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

#include "core/task/variant.h"
#include "core/utilities/typedefs.h"

namespace legate {

class TaskContext;

using VariantImpl = void (*)(TaskContext&);

namespace detail {

const char* task_name(const std::type_info&);

void task_wrapper(
  VariantImpl, const std::type_info&, const void*, size_t, const void*, size_t, Legion::Processor);

};  // namespace detail

template <typename T>
class LegateTask {
 protected:
  // Helper class for checking for various kinds of variants
  using __no  = int8_t[1];
  using __yes = int8_t[2];
  struct HasCPUVariant {
    template <typename U>
    static __yes& test(decltype(&U::cpu_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };
  struct HasOMPVariant {
    template <typename U>
    static __yes& test(decltype(&U::omp_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };
  struct HasGPUVariant {
    template <typename U>
    static __yes& test(decltype(&U::gpu_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };

 public:
  // Task wrappers so we can instrument all Legate tasks if we want
  template <VariantImpl VARIANT_IMPL>
  static void legate_task_wrapper(
    const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
  {
    detail::task_wrapper(VARIANT_IMPL, typeid(T), args, arglen, userdata, userlen, p);
  }

 public:
  // Methods for registering variants
  template <VariantImpl TASK_PTR>
  static void register_variant(Legion::ExecutionConstraintSet& execution_constraints,
                               Legion::TaskLayoutConstraintSet& layout_constraints,
                               LegateVariantCode var,
                               Legion::Processor::Kind kind,
                               const VariantOptions& options)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    Legion::CodeDescriptor desc(legate_task_wrapper<TASK_PTR>);
    auto task_id = T::TASK_ID;

    T::Registrar::record_variant(task_id,
                                 detail::task_name(typeid(T)),
                                 desc,
                                 execution_constraints,
                                 layout_constraints,
                                 var,
                                 kind,
                                 options);
  }

 public:
  static void register_variants(
    const std::map<LegateVariantCode, VariantOptions>& all_options = {});
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterCPUVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::cpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_CPU_VARIANT,
                                                    Legion::Processor::LOC_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterCPUVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_OPENMP>
class RegisterOMPVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::omp_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_OMP_VARIANT,
                                                    Legion::Processor::OMP_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterOMPVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_GPU>
class RegisterGPUVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::gpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_GPU_VARIANT,
                                                    Legion::Processor::TOC_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterGPUVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  // Make a copy of the map of options so that we can do find-or-create on it
  auto all_options_copy = all_options;
  RegisterCPUVariant<T, LegateTask<T>, HasCPUVariant::value>::register_variant(
    all_options_copy[LEGATE_CPU_VARIANT]);
  RegisterOMPVariant<T, LegateTask<T>, HasOMPVariant::value>::register_variant(
    all_options_copy[LEGATE_OMP_VARIANT]);
  RegisterGPUVariant<T, LegateTask<T>, HasGPUVariant::value>::register_variant(
    all_options_copy[LEGATE_GPU_VARIANT]);
}

}  // namespace legate
