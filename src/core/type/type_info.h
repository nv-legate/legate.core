/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "core/legate_c.h"

namespace legate {

enum class Type : int32_t {
  BOOL        = BOOL_LT,
  INT8        = INT8_LT,
  INT16       = INT16_LT,
  INT32       = INT32_LT,
  INT64       = INT64_LT,
  UINT8       = UINT8_LT,
  UINT16      = UINT16_LT,
  UINT32      = UINT32_LT,
  UINT64      = UINT64_LT,
  FLOAT16     = FLOAT16_LT,
  FLOAT32     = FLOAT32_LT,
  FLOAT64     = FLOAT64_LT,
  COMPLEX64   = COMPLEX64_LT,
  COMPLEX128  = COMPLEX128_LT,
  FIXED_ARRAY = FIXED_ARRAY_LT,
  STRUCT      = STRUCT_LT,
  STRING      = STRING_LT,
  INVALID     = INVALID_LT,
};

}  // namespace legate
