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

from typing import Any

import numpy as np

from ._legion.util import BufferBuilder

class ReductionOp:
    ADD: int
    SUB: int
    MUL: int
    DIV: int
    MAX: int
    MIN: int
    OR: int
    AND: int
    XOR: int

class Dtype:
    @staticmethod
    def fixed_array_type(element_type: Dtype, N: int) -> Dtype: ...
    @staticmethod
    def struct_type(field_types: list[Dtype], align: bool) -> Dtype: ...
    @property
    def code(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def uid(self) -> int: ...
    @property
    def variable_size(self) -> bool: ...
    def reduction_op_id(self, op_kind: int) -> int: ...
    def __repr__(self) -> str: ...
    def to_numpy_dtype(self) -> np.dtype[Any]: ...
    def serialize(self, buf: BufferBuilder) -> None: ...

class FixedArrayDtype(Dtype):
    def num_elements(self) -> int: ...
    def element_type(self) -> Dtype: ...

class StructDtype(Dtype):
    def num_fields(self) -> int: ...
    def field_type(self, field_idx: int) -> Dtype: ...

bool_: Dtype
int8: Dtype
int16: Dtype
int32: Dtype
int64: Dtype
uint8: Dtype
uint16: Dtype
uint32: Dtype
uint64: Dtype
float16: Dtype
float32: Dtype
float64: Dtype
complex64: Dtype
complex128: Dtype
string: Dtype

def array_type(element_type: Dtype, N: int) -> FixedArrayDtype: ...
def struct_type(
    field_types: list[Dtype], align: bool = False
) -> StructDtype: ...
