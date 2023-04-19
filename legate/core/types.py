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

from enum import IntEnum, unique
from typing import Any, Iterable, Type, Union

import numpy as np
import pyarrow as pa

import legate.core._lib.types as typecodes  # type: ignore[import]

from . import legion

DTType = Union[Type[bool], pa.lib.DataType]


class Complex64Dtype(pa.ExtensionType):
    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.binary(8), "complex64")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.lib.DataType, serialized: str
    ) -> Complex64Dtype:
        return Complex64Dtype()

    def __hash__(self) -> int:
        return hash(self.__class__)

    def to_pandas_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.complex64)


class Complex128Dtype(pa.ExtensionType):
    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.binary(16), "complex128")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.lib.DataType, serialized: str
    ) -> Complex128Dtype:
        return Complex128Dtype()

    def __hash__(self) -> int:
        return hash(self.__class__)

    def to_pandas_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.complex128)


bool_ = pa.bool_()
int8 = pa.int8()
int16 = pa.int16()
int32 = pa.int32()
int64 = pa.int64()
uint8 = pa.uint8()
uint16 = pa.uint16()
uint32 = pa.uint32()
uint64 = pa.uint64()
float16 = pa.float16()
float32 = pa.float32()
float64 = pa.float64()
complex64 = Complex64Dtype()
complex128 = Complex128Dtype()
string = pa.string()


class Dtype:
    def __init__(self, dtype: Any, size_in_bytes: int, code: int) -> None:
        self._dtype = dtype
        self._size_in_bytes = size_in_bytes
        self._code = code
        self._redop_ids: dict[int, int] = {}

    @property
    def type(self) -> Any:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size_in_bytes

    @property
    def code(self) -> int:
        return self._code

    @property
    def variable_size(self) -> bool:
        return self._size_in_bytes < 0

    def reduction_op_id(self, op: int) -> int:
        if op not in self._redop_ids:
            raise KeyError(
                f"{str(op)} is not a valid reduction op for type {self}"
            )
        return self._redop_ids[op]

    def register_reduction_op(self, op: int, redop_id: int) -> None:
        if op in self._redop_ids:
            raise KeyError(
                f"reduction op {str(op)} is already registered to type {self}"
            )
        self._redop_ids[op] = redop_id

    def copy_all_reduction_ops(self, other: Dtype) -> None:
        for op, redop_id in self._redop_ids.items():
            other.register_reduction_op(op, redop_id)

    def __hash__(self) -> int:
        return hash(self._dtype)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dtype):
            return (
                self._dtype == other._dtype
                and self._size_in_bytes == other._size_in_bytes
                and self._code == other._code
            )
        else:
            return self._dtype == other

    def __str__(self) -> str:
        return str(self._dtype)

    def __repr__(self) -> str:
        return f"Dtype({self._dtype}, {self.size}, {self.code})"


@unique
class ReductionOp(IntEnum):
    ADD = typecodes.ADD
    SUB = typecodes.SUB
    MUL = typecodes.MUL
    DIV = typecodes.DIV
    MAX = typecodes.MAX
    MIN = typecodes.MIN
    OR = typecodes.OR
    AND = typecodes.AND
    XOR = typecodes.XOR


# TODO: These should use the enum in legate_c.h

_CORE_DTYPES = [
    Dtype(bool, 1, typecodes.BOOL),
    Dtype(bool_, 1, typecodes.BOOL),
    Dtype(int8, 1, typecodes.INT8),
    Dtype(int16, 2, typecodes.INT16),
    Dtype(int32, 4, typecodes.INT32),
    Dtype(int64, 8, typecodes.INT64),
    Dtype(uint8, 1, typecodes.UINT8),
    Dtype(uint16, 2, typecodes.UINT16),
    Dtype(uint32, 4, typecodes.UINT32),
    Dtype(uint64, 8, typecodes.UINT64),
    Dtype(float16, 2, typecodes.FLOAT16),
    Dtype(float32, 4, typecodes.FLOAT32),
    Dtype(float64, 8, typecodes.FLOAT64),
    Dtype(complex64, 8, typecodes.COMPLEX64),
    Dtype(complex128, 16, typecodes.COMPLEX128),
    Dtype(string, -1, typecodes.STRING),
]


_CORE_DTYPE_MAP = dict([(dtype.type, dtype) for dtype in _CORE_DTYPES])


def _register_reduction_ops(
    dtypes: list[Dtype], ops: Iterable[ReductionOp]
) -> None:
    for dtype in dtypes:
        for op in ops:
            redop_id = (
                legion.LEGION_REDOP_BASE
                + op.value * legion.LEGION_TYPE_TOTAL
                + dtype.code
            )
            dtype.register_reduction_op(op, redop_id)


_redops_float = (
    ReductionOp.ADD,
    ReductionOp.SUB,
    ReductionOp.MUL,
    ReductionOp.DIV,
    ReductionOp.MIN,
    ReductionOp.MAX,
)

_register_reduction_ops(_CORE_DTYPES[:10], ReductionOp)
_register_reduction_ops(_CORE_DTYPES[10:13], _redops_float)
_register_reduction_ops(_CORE_DTYPES[13:-1], _redops_float[:4])


class TypeSystem:
    def __init__(self, inherit_core_types: bool = True) -> None:
        self._types = _CORE_DTYPE_MAP.copy() if inherit_core_types else {}

    # ty should hashable
    def __contains__(self, ty: Any) -> bool:
        return ty in self._types

    def __getitem__(self, ty: Any) -> Dtype:
        if ty not in self._types:
            raise KeyError(f"{ty} is not a valid type in this type system")
        return self._types[ty]

    def add_type(self, ty: Any, size_in_bytes: int, code: int) -> Dtype:
        if ty in self._types:
            raise KeyError(f"{ty} is already in this type system")
        dtype = Dtype(ty, size_in_bytes, code)
        self._types[ty] = dtype
        return dtype

    def make_alias(
        self, alias: Any, src_type: Any, copy_reduction_ops: bool = True
    ) -> Dtype:
        dtype = self[src_type]
        copy = Dtype(alias, dtype.size, dtype.code)
        if copy_reduction_ops:
            dtype.copy_all_reduction_ops(copy)
        self._types[alias] = copy
        return copy

    def __str__(self) -> str:
        return str(self._types)
