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

#include "core/task/task.h"

namespace legate {

namespace detail {

std::string generate_task_name(const std::type_info&);

void task_wrapper(
  VariantImpl, const char*, const void*, size_t, const void*, size_t, Legion::Processor);

};  // namespace detail

template <typename T>
template <VariantImpl VARIANT_IMPL>
/*static*/ void LegateTask<T>::legate_task_wrapper(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  detail::task_wrapper(VARIANT_IMPL, task_name(), args, arglen, userdata, userlen, p);
}

template <typename T>
template <VariantImpl VARIANT_IMPL>
/*static*/ void LegateTask<T>::register_variant(
  Legion::ExecutionConstraintSet& execution_constraints,
  Legion::TaskLayoutConstraintSet& layout_constraints,
  LegateVariantCode var,
  Legion::Processor::Kind kind,
  const VariantOptions& options)
{
  // Construct the code descriptor for this task so that the library
  // can register it later when it is ready
  Legion::CodeDescriptor desc(legate_task_wrapper<VARIANT_IMPL>);
  auto task_id = T::TASK_ID;

  T::Registrar::record_variant(
    task_id, task_name(), desc, execution_constraints, layout_constraints, var, kind, options);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  // Make a copy of the map of options so that we can do find-or-create on it
  auto all_options_copy = all_options;
  detail::RegisterVariant<T, detail::CPUVariant>::register_variant(
    all_options_copy[LEGATE_CPU_VARIANT]);
  detail::RegisterVariant<T, detail::OMPVariant>::register_variant(
    all_options_copy[LEGATE_OMP_VARIANT]);
  detail::RegisterVariant<T, detail::GPUVariant>::register_variant(
    all_options_copy[LEGATE_GPU_VARIANT]);
}

template <typename T>
/*static*/ const char* LegateTask<T>::task_name()
{
  static std::string result = detail::generate_task_name(typeid(T));
  return result.c_str();
}

}  // namespace legate
