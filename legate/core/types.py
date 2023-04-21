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
from __future__ import annotations

from enum import IntEnum, unique

import legate.core._lib.types as ext  # type: ignore[import]


@unique
class ReductionOp(IntEnum):
    ADD = ext.ADD
    SUB = ext.SUB
    MUL = ext.MUL
    DIV = ext.DIV
    MAX = ext.MAX
    MIN = ext.MIN
    OR = ext.OR
    AND = ext.AND
    XOR = ext.XOR


Dtype = ext.Dtype
FixedArrayDtype = ext.FixedArrayDtype
StructDtype = ext.StructDtype


bool_ = Dtype.primitive_type(ext.BOOL)
int8 = Dtype.primitive_type(ext.INT8)
int16 = Dtype.primitive_type(ext.INT16)
int32 = Dtype.primitive_type(ext.INT32)
int64 = Dtype.primitive_type(ext.INT64)
uint8 = Dtype.primitive_type(ext.UINT8)
uint16 = Dtype.primitive_type(ext.UINT16)
uint32 = Dtype.primitive_type(ext.UINT32)
uint64 = Dtype.primitive_type(ext.UINT64)
float16 = Dtype.primitive_type(ext.FLOAT16)
float32 = Dtype.primitive_type(ext.FLOAT32)
float64 = Dtype.primitive_type(ext.FLOAT64)
complex64 = Dtype.primitive_type(ext.COMPLEX64)
complex128 = Dtype.primitive_type(ext.COMPLEX128)
string = Dtype.string_type()


def array_type(element_type: Dtype, N: int) -> FixedArrayDtype:
    return Dtype.fixed_array_type(element_type, N)


def struct_type(field_types: list[Dtype], align: bool = False) -> StructDtype:
    return Dtype.struct_type(field_types, align)
