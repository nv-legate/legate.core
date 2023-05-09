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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import FieldSpace


class FieldID:
    def __init__(self, field_space: FieldSpace, fid: int, type: Any) -> None:
        """
        A FieldID class wraps a `legion_field_id_t` in the Legion C API.
        It provides a canonical way to represent an allocated field in a
        field space and means by which to deallocate the field.

        Parameters
        ----------
        field_space : FieldSpace
            The owner field space for this field
        fid : int
            The ID for this field
        type : type
            The type of this field
        """
        self.field_space = field_space
        self._type = type
        self.field_id = fid

    def destroy(self, unordered: bool = False) -> None:
        """
        Deallocate this field from the field space
        """
        self.field_space.destroy_field(self.field_id, unordered)

    @property
    def fid(self) -> int:
        return self.field_id

    @property
    def type(self) -> Any:
        return self._type
