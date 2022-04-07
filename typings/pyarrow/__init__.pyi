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

from typing import Any, Union

from .lib import (
    DataType,
    binary,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

class Field:
    name: str
    type: DataType
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def with_name(self, name: str) -> Field: ...

def field(
    name: Union[str, bytes],
    type: DataType,
    nullable: bool = True,
    metadata: Any = None,
) -> Field: ...

class Schema:
    types: Any
    def field(self, i: Union[str, int]) -> Field: ...
    def get_all_field_indices(self, name: str) -> list[int]: ...
    def get_field_index(self, name: str) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Field: ...

def schema(fields: Any, metadata: Any = None) -> Schema: ...

class ExtensionType:
    def __init__(self, dtype: DataType, name: str) -> None: ...

__all__ = (
    "binary",
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "Field",
    "Schema",
    "DataType",
)
