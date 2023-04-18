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
from __future__ import annotations

import os
from typing import Any, Union

from ..install_info import header, libpath  # type: ignore [import]
from .legate import Library

# This is annoying but install_info is not present on unbuilt source, but is
# present in built source. So we either get an unfollowed-import error, or an
# unused-ignore error. Allow unused-ignores just in this file to work around
# mypy: warn-unused-ignores=False


class CoreLib(Library):
    def __init__(self) -> None:
        super().__init__()
        self._lib: Union[Any, None] = None

    def get_name(self) -> str:
        return "legate.core"

    def get_shared_library(self) -> str:
        libname = "liblgcore" + self.get_library_extension()
        return os.path.join(libpath, libname)

    def get_c_header(self) -> str:
        return header

    def initialize(self, shared_lib: Any) -> None:
        self._lib = shared_lib
        shared_lib.legate_parse_config()

    def get_registration_callback(self) -> str:
        return "legate_core_perform_registration"

    def destroy(self) -> None:
        if not self._lib:
            raise RuntimeError("CoreLib was never initialized")
        self._lib.legate_shutdown()


core_library = CoreLib()
