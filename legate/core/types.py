# Copyright 2021 NVIDIA Corporation
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


import pyarrow as pa

from legion_cffi import lib as legion


class Complex64Dtype(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.binary(8), "complex64")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(self, storage_type, serialized):
        return Complex64Dtype()

    def __hash__(self):
        return hash(self.__class__)


class Complex128Dtype(pa.ExtensionType):
    def __init__(self):
        pa.ExtensionType.__init__(self, pa.binary(16), "complex128")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(self, storage_type, serialized):
        return Complex128Dtype()

    def __hash__(self):
        return hash(self.__class__)


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


class _Dtype(object):
    def __init__(self, dtype, size_in_bytes, code):
        self._dtype = dtype
        self._size_in_bytes = size_in_bytes
        self._code = code

    @property
    def type(self):
        return self._dtype

    @property
    def size(self):
        return self._size_in_bytes

    @property
    def code(self):
        return self._code

    def __hash__(self):
        return hash(self._dtype)

    def __str__(self):
        return str(self._dtype)

    def __repr__(self):
        return f"Dtype({self._dtype}, {self.code}, {self.size})"


_CORE_DTYPES = [
    _Dtype(bool, 1, legion.LEGION_TYPE_BOOL),
    _Dtype(bool_, 1, legion.LEGION_TYPE_BOOL),
    _Dtype(int8, 1, legion.LEGION_TYPE_INT8),
    _Dtype(int16, 2, legion.LEGION_TYPE_INT16),
    _Dtype(int32, 4, legion.LEGION_TYPE_INT32),
    _Dtype(int64, 8, legion.LEGION_TYPE_INT64),
    _Dtype(uint8, 1, legion.LEGION_TYPE_UINT8),
    _Dtype(uint16, 2, legion.LEGION_TYPE_UINT16),
    _Dtype(uint32, 4, legion.LEGION_TYPE_UINT32),
    _Dtype(uint64, 8, legion.LEGION_TYPE_UINT64),
    _Dtype(float16, 2, legion.LEGION_TYPE_FLOAT16),
    _Dtype(float32, 4, legion.LEGION_TYPE_FLOAT32),
    _Dtype(float64, 8, legion.LEGION_TYPE_FLOAT64),
    _Dtype(complex64, 8, legion.LEGION_TYPE_COMPLEX64),
    _Dtype(complex128, 16, legion.LEGION_TYPE_COMPLEX128),
]


_CORE_DTYPE_MAP = dict([(dtype.type, dtype) for dtype in _CORE_DTYPES])


class TypeSystem(object):
    def __init__(self, inherit_core_types=True):
        self._types = _CORE_DTYPE_MAP.copy() if inherit_core_types else {}

    def __contains__(self, ty):
        return ty in self._types

    def __getitem__(self, ty):
        if ty not in self._types:
            raise KeyError(f"{ty} is not a valid type in this type system")
        return self._types[ty]

    def add_type(self, ty, size_in_bytes, code):
        if ty in self._types:
            raise KeyError(f"{ty} is already in this type system")
        dtype = _Dtype(ty, size_in_bytes, code)
        self._types[dtype] = dtype

    def make_alias(self, alias, src_type):
        dtype = self[src_type]
        copy = _Dtype(alias, dtype.size, dtype.code)
        self._types[alias] = copy

    def __str__(self):
        return str(self._types)
