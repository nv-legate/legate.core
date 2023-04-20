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


from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

import cython
import numpy as np


cdef extern from "core/legate_c.h" nogil:
    ctypedef enum legate_core_type_code_t:
        _BOOL "BOOL_LT"
        _INT8 "INT8_LT"
        _INT16 "INT16_LT"
        _INT32 "INT32_LT"
        _INT64 "INT64_LT"
        _UINT8 "UINT8_LT"
        _UINT16 "UINT16_LT"
        _UINT32 "UINT32_LT"
        _UINT64 "UINT64_LT"
        _FLOAT16 "FLOAT16_LT"
        _FLOAT32 "FLOAT32_LT"
        _FLOAT64 "FLOAT64_LT"
        _COMPLEX64 "COMPLEX64_LT"
        _COMPLEX128 "COMPLEX128_LT"
        _FIXED_ARRAY "FIXED_ARRAY_LT"
        _STRUCT "STRUCT_LT"
        _STRING "STRING_LT"
        _INVALID "INVALID_LT"

    ctypedef enum legate_core_reduction_op_kind_t:
        _ADD "ADD_LT"
        _SUB "SUB_LT"
        _MUL "MUL_LT"
        _DIV "DIV_LT"
        _MAX "MAX_LT"
        _MIN "MIN_LT"
        _OR  "OR_LT"
        _AND "AND_LT"
        _XOR "XOR_LT"

BOOL        = legate_core_type_code_t._BOOL
INT8        = legate_core_type_code_t._INT8
INT16       = legate_core_type_code_t._INT16
INT32       = legate_core_type_code_t._INT32
INT64       = legate_core_type_code_t._INT64
UINT8       = legate_core_type_code_t._UINT8
UINT16      = legate_core_type_code_t._UINT16
UINT32      = legate_core_type_code_t._UINT32
UINT64      = legate_core_type_code_t._UINT64
FLOAT16     = legate_core_type_code_t._FLOAT16
FLOAT32     = legate_core_type_code_t._FLOAT32
FLOAT64     = legate_core_type_code_t._FLOAT64
COMPLEX64   = legate_core_type_code_t._COMPLEX64
COMPLEX128  = legate_core_type_code_t._COMPLEX128
FIXED_ARRAY = legate_core_type_code_t._FIXED_ARRAY
STRUCT      = legate_core_type_code_t._STRUCT
STRING      = legate_core_type_code_t._STRING
INVALID     = legate_core_type_code_t._INVALID

ADD = legate_core_reduction_op_kind_t._ADD
SUB = legate_core_reduction_op_kind_t._SUB
MUL = legate_core_reduction_op_kind_t._MUL
DIV = legate_core_reduction_op_kind_t._DIV
MAX = legate_core_reduction_op_kind_t._MAX
MIN = legate_core_reduction_op_kind_t._MIN
OR  = legate_core_reduction_op_kind_t._OR
AND = legate_core_reduction_op_kind_t._AND
XOR = legate_core_reduction_op_kind_t._XOR

_NUMPY_DTYPES = {
    BOOL : np.dtype(np.bool_),
    INT8 : np.dtype(np.int8),
    INT16 : np.dtype(np.int16),
    INT32 : np.dtype(np.int32),
    INT64 : np.dtype(np.int64),
    UINT8 : np.dtype(np.uint8),
    UINT16 : np.dtype(np.uint16),
    UINT32 : np.dtype(np.uint32),
    UINT64 : np.dtype(np.uint64),
    FLOAT16 : np.dtype(np.float16),
    FLOAT32 : np.dtype(np.float32),
    FLOAT64 : np.dtype(np.float64),
    COMPLEX64 : np.dtype(np.complex64),
    COMPLEX128 : np.dtype(np.complex128),
    STRING : np.dtype(np.str_),
}


cdef extern from "core/type/type_info.h" namespace "legate" nogil:
    cdef cppclass Type:
        ctypedef enum Code:
            pass
        int code
        unsigned int size()
        unsigned int alignment()
        int uid()
        bool variable_size()
        unique_ptr[Type] clone()
        string to_string()
        int find_reduction_operator(int) except+

    cdef cppclass FixedArrayType(Type):
        unsigned int num_elements()
        const Type* element_type()

    cdef cppclass StructType(Type):
        unsigned int num_fields()
        const Type* field_type(unsigned int)
        bool aligned()

    cdef unique_ptr[Type] primitive_type(int code)

    cdef unique_ptr[Type] string_type()

    cdef unique_ptr[Type] fixed_array_type(
        unique_ptr[Type] element_type, unsigned int N
    ) except+

    cdef unique_ptr[Type] struct_type_raw_ptrs(
        vector[Type*] field_types, bool
    ) except+


cdef class DataType:
    cdef unique_ptr[Type] _type

    @staticmethod
    cdef DataType from_ptr(Type* ty):
        cdef DataType dtype = DataType.__new__(DataType)
        dtype._type.reset(ty)
        return dtype

    @staticmethod
    def primitive_type(int code) -> DataType:
        return DataType.from_ptr(primitive_type(<Type.Code> code).release())

    @staticmethod
    def string_type() -> DataType:
        return DataType.from_ptr(string_type().release())

    @staticmethod
    def fixed_array_type(DataType element_type, unsigned N) -> DataType:
        return DataType.from_ptr(
            fixed_array_type(element_type._type.get().clone(), N).release()
        )

    @staticmethod
    def struct_type(list field_types, bool align) -> DataType:
        cdef vector[Type*] types
        for field_type in field_types:
            types.push_back(
                cython.cast(DataType, field_type)._type.get().clone().release()
            )
        return DataType.from_ptr(struct_type_raw_ptrs(types, align).release())

    @property
    def code(self) -> int:
        return <int> self._type.get().code

    @property
    def size(self) -> int:
        return self._type.get().size()

    @property
    def alignment(self) -> int:
        return self._type.get().alignment()

    @property
    def uid(self) -> int:
        return self._type.get().uid()

    @property
    def variable_size(self) -> bool:
        return self._type.get().variable_size()

    def reduction_op_id(self, int op_kind) -> int:
        return self._type.get().find_reduction_operator(op_kind)

    def __repr__(self) -> str:
        return self._type.get().to_string().decode()

    def num_elements(self) -> int:
        if self.code != FIXED_ARRAY:
            raise ValueError(
                "`num_elements` is defined only for a fixed array type"
            )
        cdef FixedArrayType* ptr = <FixedArrayType*> self._type.get()
        return ptr.num_elements()

    def element_type(self) -> DataType:
        if self.code != FIXED_ARRAY:
            raise ValueError(
                "`element_type` is defined only for a fixed array type"
            )
        cdef FixedArrayType* ptr = <FixedArrayType*> self._type.get()
        return DataType.from_ptr(ptr.element_type().clone().release())

    def num_fields(self) -> int:
        if self.code != STRUCT:
            raise ValueError(
                "`num_fields` is defined only for a struct type"
            )
        cdef StructType* ptr = <StructType*> self._type.get()
        return ptr.num_fields()

    def field_type(self, int field_idx) -> DataType:
        if self.code != STRUCT:
            raise ValueError(
                "`field_type` is defined only for a struct type"
            )
        cdef StructType* ptr = <StructType*> self._type.get()
        return DataType.from_ptr(ptr.field_type(field_idx).clone().release())

    def aligned(self) -> bool:
        if self.code != STRUCT:
            raise ValueError(
                "`aligned` is defined only for a struct type"
            )
        cdef StructType* ptr = <StructType*> self._type.get()
        return ptr.aligned()

    def to_numpy_dtype(self):
        code = self.code
        if code in _NUMPY_DTYPES:
            return _NUMPY_DTYPES[code]
        elif code == FIXED_ARRAY:
            arr_type = (
                self.element_type().to_numpy_dtype(), self.num_elements()
            )
            # Return a singleton struct type, as NumPy would flatten away
            # nested arrays
            return np.dtype({"names": ("_0",), "formats": (arr_type,)})
        elif code == STRUCT:
            num_fields = self.num_fields()
            names = tuple(
                f"_{field_idx}" for field_idx in range(num_fields)
            )
            formats = tuple(
                self.field_type(field_idx).to_numpy_dtype()
                for field_idx in range(num_fields)
            )
            return np.dtype(
                {"names": names, "formats": formats}, align=self.aligned()
            )

        else:
            raise ValueError(f"Invalid type code: {code}")

    def serialize(self, buf) -> None:
        code = self.code
        buf.pack_32bit_int(code)
        if code == FIXED_ARRAY:
            buf.pack_32bit_int(self.uid)
            buf.pack_32bit_int(self.num_elements())
            self.element_type().serialize(buf)
        elif code == STRUCT:
            num_fields = self.num_fields()
            buf.pack_32bit_int(self.uid)
            buf.pack_32bit_int(num_fields)
            for field_idx in range(num_fields):
                self.field_type(field_idx).serialize(buf)
