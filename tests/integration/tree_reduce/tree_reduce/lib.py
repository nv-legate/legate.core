# Copyright 2023 NVIDIA Corporation
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
from typing import Any

from legate.core import Library, get_legate_runtime


class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from tree_reduce.install_info import libpath

        return os.path.join(
            libpath, f"libtree_reduce{self.get_library_extension()}"
        )

    def get_c_header(self) -> str:
        from tree_reduce.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass


user_lib = UserLibrary("tree_reduce")
user_context = get_legate_runtime().register_library(user_lib)
