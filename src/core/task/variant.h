/* Copyright 2023 NVIDIA Corporation
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

#include "core/utilities/typedefs.h"

namespace legate {

// We're going to allow for each task to use only up to 341 scalar output stores
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

struct VariantOptions {
  bool leaf{true};
  bool inner{false};
  bool idempotent{false};
  bool concurrent{false};
  size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  VariantOptions& with_leaf(bool leaf);
  VariantOptions& with_inner(bool inner);
  VariantOptions& with_idempotent(bool idempotent);
  VariantOptions& with_concurrent(bool concurrent);
  VariantOptions& with_return_size(size_t return_size);
};

namespace detail {

template <typename T>
using void_t = void;

template <typename T, typename = void>
struct CPUVariant : std::false_type {};

template <typename T, typename = void>
struct OMPVariant : std::false_type {};

template <typename T, typename = void>
struct GPUVariant : std::false_type {};

template <typename T>
struct CPUVariant<T, void_t<decltype(T::cpu_variant)>> : std::true_type {
  static constexpr auto variant   = T::cpu_variant;
  static constexpr auto id        = LEGATE_CPU_VARIANT;
  static constexpr auto proc_kind = Processor::LOC_PROC;
};

template <typename T>
struct OMPVariant<T, void_t<decltype(T::omp_variant)>> : std::true_type {
  static constexpr auto variant   = T::omp_variant;
  static constexpr auto id        = LEGATE_OMP_VARIANT;
  static constexpr auto proc_kind = Processor::OMP_PROC;
};

template <typename T>
struct GPUVariant<T, void_t<decltype(T::gpu_variant)>> : std::true_type {
  static constexpr auto variant   = T::gpu_variant;
  static constexpr auto id        = LEGATE_GPU_VARIANT;
  static constexpr auto proc_kind = Processor::TOC_PROC;
};

template <typename T, template <typename...> typename SELECTOR, bool HAS_VARIANT>
struct RegisterVariantImpl {
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::BASE::template register_variant<SELECTOR<T>::variant>(
      execution_constraints, layout_constraints, SELECTOR<T>::id, SELECTOR<T>::proc_kind, options);
  }
};

template <typename T, template <typename...> typename SELECTOR>
struct RegisterVariantImpl<T, SELECTOR, false> {
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, template <typename...> typename SELECTOR>
struct RegisterVariant {
  static void register_variant(const VariantOptions& options)
  {
    RegisterVariantImpl<T, SELECTOR, SELECTOR<T>::value>::register_variant(options);
  }
};

}  // namespace detail

}  // namespace legate
