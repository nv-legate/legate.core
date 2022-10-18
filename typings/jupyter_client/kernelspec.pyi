# Copyright 2022 NVIDIA Corporation
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

from typing import Any

class KernelSpec:
    display_name: str
    metadata: dict[str, Any]

    def __init__(
        self,
        argv: list[str],
        env: dict[str, str],
        display_name: str,
        language: str,
        metadata: dict[str, Any],
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class NoSuchKernel(Exception): ...

class KernelSpecManager:
    def __init__(self, **kwargs: Any) -> None: ...
    def get_kernel_spec(self, kernel_name: str) -> KernelSpec: ...
    def install_kernel_spec(
        self, source_dir: str, kernel_name: str, user: bool, prefix: str | None
    ) -> None: ...
