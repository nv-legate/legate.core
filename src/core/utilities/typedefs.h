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

/**
 * @file
 * @brief Type aliases to Legion components
 */

namespace legate {

/**
 * @brief Function signature for task variants. Each task variant must be a function of this type.
 */
class TaskContext;
using VariantImpl = void (*)(TaskContext&);

// C enum typedefs
using LegateVariantCode = legate_core_variant_t;
using LegateMappingTag  = legate_core_mapping_tag_t;

using Logger = Legion::Logger;

extern Logger log_legate;

// Re-export Legion types

using TunableID = Legion::TunableID;

// Geometry types

/** @defgroup geometry Geometry types
 *
 * @{
 */

/**
 * @brief Coordinate type.
 */
using coord_t = Legion::coord_t;

/**
 * @brief Type for multi-dimensional points.
 *
 * Point objects support index expressions; they can be accessed like a statically-sized array.
 * Point objects also support usual arithmetic operators and a dot opreator.
 *
 * For a complete definition, see
 * [`Realm::Point`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L46-L124).
 */
template <int DIM, typename T = coord_t>
using Point = Legion::Point<DIM, T>;

/**
 * @brief Type for multi-dimensional rectangles.
 *
 * Each rectangle consists of two legate::Point objects, one for the lower
 * bounds (`.lo`) and one for the upper bounds (`.hi`).
 *
 * For a complete definition, see
 * [`Realm::Rect`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L126-L212).
 */
template <int DIM, typename T = coord_t>
using Rect = Legion::Rect<DIM, T>;

/**
 * @brief Dimension-erased type for multi-dimensional points.
 *
 * For a complete definition, see
 * [`Legion::DomainPoint`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L127-L253).
 */
using DomainPoint = Legion::DomainPoint;

/**
 * @brief Dimension-erased type for multi-dimensional rectangles.
 *
 * For a complete definition, see
 * [`Legion::Domain`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L255-L543).
 */
using Domain = Legion::Domain;

/** @} */  // end of geometry

// Accessor types

/** @defgroup accessor Accessor types
 *
 * Accessors provide an interface to access values in stores. Access modes are encoded
 * in the accessor types so that the compiler can catch invalid accesses. Accessors also
 * provide bounds checks (which can be turned on with a compile flag).
 *
 * All accessors have a `ptr` method that returns a raw pointer to the underlying allocation.
 * The caller can optionally pass an array to query strides of dimensions, necessary for correct
 * accesse. Unlike the accesses mediated by accessors, raw pointer accesses are not protected by
 * Legate, and thus the developer should make sure of safety of the accesses.
 *
 * The most common mistake with raw pointers from reduction accessors are that the code overwrites
 * values to the elements, instead of reducing them. The key contract with reduction is that
 * the values must be reduced to the elements in the store. So, any client code that uses a raw
 * pointer to a reduction store should make sure that it makes updates to the effect of reducing
 * its contributions to the original elements. Not abiding by this contract can lead to
 * non-deterministic conrrectness issues.
 *
 * @{
 */

/**
 * @brief Read-only accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2555-L2562)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Write-only accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2575-L2581)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Read-write accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2564-L2573)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Reduction accessor
 *
 * Unlike the other accessors, an index expression on a reduction accessor allows the client to
 * perform only two operations, `<<=` and `reduce`, both of which reduce a value to the chosen
 * element.
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2837-L2848)
 * for details about the reduction accessor.
 */
template <typename REDOP, bool EXCLUSIVE, int N, typename T = coord_t>
using AccessorRD = Legion::
  ReductionAccessor<REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>>;

/** @} */  // end of accessor

// Iterators

/** @defgroup iterator Iterator types
 *
 * @{
 */

/**
 * @brief Iterator that iterates all points in a given `legate::Rect`.
 *
 * See
 * [Realm::PointInRectIterator](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L239-L255)
 * for a complete definition.
 */
template <int DIM, typename T = coord_t>
using PointInRectIterator = Legion::PointInRectIterator<DIM, T>;

/**
 * @brief Iterator that iterates all points in a given `legate::Domain`.
 *
 * See
 * [Legion::PointInDomainIterator](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L599-L622)
 * for a complete definition.
 */
template <int DIM, typename T = coord_t>
using PointInDomainIterator = Legion::PointInDomainIterator<DIM, T>;

/** @} */  // end of iterator

// Machine

/** @defgroup machine Machine objects
 *
 * @{
 */

/**
 * @brief Logical processor handle
 *
 * Legate libraries rarely use processor handles directly and there are no Legate APIs that take
 * a processor handle. However, the libraries may want to query the processor that runs the
 * current task to perform some processor- or processor kind-specific operations. In that case,
 * `legate::Processor::get_executing_processor` can be used. Other useful memobers of
 * `legate::Processor` are the `kind` method, which returns the processor kind, and
 * `legate::Processor::Kind`, an enum for all processor types.
 *
 * See
 * [`Realm::Processor`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/processor.h#L35-L141)
 * for a complete definition. The list of processor types can be found
 * [here](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/realm_c.h#L45-L54).
 *
 */
using Processor = Legion::Processor;

/**
 * @brief Logical memory handle
 *
 * In Legate, libraries will never have to use memory handles directly. However, some Legate
 * APIs (e.g., legate::create_buffer) take a memory kind as an argument; `legate::Memory::Kind`
 * is an enum for all memory types.
 *
 * See
 * [`Realm::Memory`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/memory.h#L30-L65)
 * for a complete definition. The list of memory types can be found
 * [here](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/realm_c.h#L63-L78).
 */
using Memory = Legion::Memory;

/** @} */  // end of machine

// Reduction operators

/** @defgroup reduction Built-in reduction operators
 *
 * All built-in operators are defined for signed and unsigned integer types. Floating point
 * types (`__half`, `float`, and `double`) are supported by all but bitwise operators. Arithmetic
 * operators also cover complex types `complex<__half>` and `complex<float>`.
 *
 * For details about reduction operators, See LibraryContext::register_reduction_operator.
 *
 * @{
 */

/**
 * @brief Reduction with addition
 *
 * See
 * [`Legion::SumReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L46-L285).
 */
template <typename T>
using SumReduction = Legion::SumReduction<T>;

/**
 * @brief Reduction with subtraction
 *
 * See
 * [`Legion::DiffReduction`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L287-L492).
 */
template <typename T>
using DiffReduction = Legion::DiffReduction<T>;

/**
 * @brief Reduction with multiplication
 *
 * See
 * [`Legion::ProdReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L494-L714).
 */
template <typename T>
using ProdReduction = Legion::ProdReduction<T>;

/**
 * @brief Reduction with division
 *
 * See
 * [`Legion::DivReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L716-L921).
 */
template <typename T>
using DivReduction = Legion::DivReduction<T>;

/**
 * @brief Reduction with the binary max operator
 *
 * See
 * [`Legion::MaxReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L923-L1109).
 */
template <typename T>
using MaxReduction = Legion::MaxReduction<T>;

/**
 * @brief Reduction with the binary min operator
 *
 * See
 * [`Legion::MinReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1111-L1297).
 */
template <typename T>
using MinReduction = Legion::MinReduction<T>;

/**
 * @brief Reduction with bitwise or
 *
 * See
 * [`Legion::OrReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1299-L1423).
 */
template <typename T>
using OrReduction = Legion::OrReduction<T>;

/**
 * @brief Reduction with bitwise and
 *
 * See
 * [`Legion::AndReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1425-L1549).
 */
template <typename T>
using AndReduction = Legion::AndReduction<T>;

/**
 * @brief Reduction with bitwise xor
 *
 * See
 * [`Legion::XorReduction`](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1551-L1690).
 */
template <typename T>
using XorReduction = Legion::XorReduction<T>;

/** @} */  // end of reduction

}  // namespace legate
