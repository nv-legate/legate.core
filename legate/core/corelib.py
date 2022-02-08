# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os

from .context import ResourceConfig
from .install_info import header, libpath
from .legate import Library


class CoreLib(Library):
    def __init__(self):
        self._lib = None

    def get_name(self):
        return "legate.core"

    def get_shared_library(self):
        libname = "liblgcore" + self.get_library_extension()
        return os.path.join(libpath, libname)

    def get_c_header(self):
        return header

    def initialize(self, shared_lib):
        self._lib = shared_lib
        shared_lib.legate_parse_config()

    def get_registration_callback(self):
        return "legate_core_perform_registration"

    def get_resource_configuration(self):
        config = ResourceConfig()
        config.max_tasks = self._lib.LEGATE_CORE_NUM_TASK_IDS
        config.max_projections = self._lib.LEGATE_CORE_MAX_FUNCTOR_ID
        config.max_shardings = self._lib.LEGATE_CORE_MAX_FUNCTOR_ID
        return config

    def destroy(self):
        self._lib.legate_shutdown()
