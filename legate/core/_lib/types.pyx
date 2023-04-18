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
