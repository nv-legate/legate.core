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
#include "legate_defines.h"

namespace legate {

// C enum typedefs
using LegateVariantCode = legate_core_variant_t;
using LegateTypeCode    = legate_core_type_code_t;
using LegateMappingTag  = legate_core_mapping_tag_t;

using Logger = Legion::Logger;

extern Logger log_legate;

// Re-export Legion types

using TunableID = Legion::TunableID;

// Geometry types

using coord_t = Legion::coord_t;

template <int DIM, typename T = coord_t>
using Point = Legion::Point<DIM, T>;
template <int DIM, typename T = coord_t>
using Rect = Legion::Rect<DIM, T>;

using Domain      = Legion::Domain;
using DomainPoint = Legion::DomainPoint;

// Accessor types

template <typename FT, int N, typename T = coord_t>
using AccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = coord_t>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = coord_t>
using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename REDOP, bool EXCLUSIVE, int N, typename T = coord_t>
using AccessorRD = Legion::
  ReductionAccessor<REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>>;

// Iterators

template <int DIM, typename T = coord_t>
using PointInRectIterator = Legion::PointInRectIterator<DIM, T>;
template <int DIM, typename T = coord_t>
using RectInDomainIterator = Legion::RectInDomainIterator<DIM, T>;
template <int DIM, typename T = coord_t>
using PointInDomainIterator = Legion::PointInDomainIterator<DIM, T>;

// Machine

using Processor = Legion::Processor;
using Memory    = Legion::Memory;

// Reduction operators

template <typename T>
using SumReduction = Legion::SumReduction<T>;
template <typename T>
using DiffReduction = Legion::DiffReduction<T>;
template <typename T>
using ProdReduction = Legion::ProdReduction<T>;
template <typename T>
using DivReduction = Legion::DivReduction<T>;
template <typename T>
using MaxReduction = Legion::MaxReduction<T>;
template <typename T>
using MinReduction = Legion::MinReduction<T>;
template <typename T>
using OrReduction = Legion::OrReduction<T>;
template <typename T>
using AndReduction = Legion::AndReduction<T>;
template <typename T>
using XorReduction = Legion::XorReduction<T>;

}  // namespace legate
