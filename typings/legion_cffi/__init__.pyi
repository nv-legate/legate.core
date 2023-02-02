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

from typing import Any, Callable

# cheating a bit here, these are part of cffi but it is a mess
class CData:
    def __getitem__(self, idx: int) -> Any: ...
    def __setitem__(self, idx: int, value: Any) -> None: ...

class CType:
    cname: str

class FFI:
    NULL: CData

    def new(self, ctype: str, *args: Any) -> CData: ...
    def cast(self, ctype: str, value: Any) -> CData: ...
    def cdef(self, cstring: str) -> None: ...
    def dlopen(self, path: str) -> Any: ...
    def typeof(self, tpy: str | CData) -> CType: ...
    def addressof(self, value: CData) -> Any: ...
    def sizeof(self, value: Any) -> int: ...
    def from_buffer(self, value: CData | memoryview) -> Any: ...
    def buffer(self, value: CData, size: int = 0) -> Any: ...
    def unpack(self, value: CData, maxlen: int = 0) -> bytes: ...
    def gc(
        self, value: CData, destructor: Callable[[CData], None], size: int = 0
    ) -> CData: ...

ffi: FFI

is_legion_python: bool
