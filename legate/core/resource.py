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

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .context import Context


class ResourceConfig:
    max_tasks = 1_000_000
    max_reduction_ops = 0
    max_mappers = 1
    max_projections = 0
    max_shardings = 0


class ResourceScope:
    def __init__(
        self, context: Context, base: Optional[int], category: str
    ) -> None:
        self._context = context
        self._base = base
        self._category = category

    @property
    def scope(self) -> str:
        return self._context._library.get_name()

    def translate(self, resource_id: int) -> int:
        if self._base is None:
            raise ValueError(f"{self.scope} has not {self._category}")
        return self._base + resource_id
