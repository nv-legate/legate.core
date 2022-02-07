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

#include "legion.h"

#include "core/legate_c.h"

namespace legate {

extern Legion::Logger log_legate;

template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename REDOP, bool EXCLUSIVE, int N, typename T = Legion::coord_t>
using AccessorRD = Legion::
  ReductionAccessor<REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T>;

using TunableID = Legion::TunableID;

// C enum typedefs
using LegateVariantCode = legate_core_variant_t;
using LegateTypeCode    = legate_core_type_code_t;
using LegateMappingTag  = legate_core_mapping_tag_t;

}  // namespace legate
