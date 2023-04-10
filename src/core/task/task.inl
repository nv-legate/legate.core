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
  VariantImpl, const std::string&, const void*, size_t, const void*, size_t, Legion::Processor);

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
/*static*/ void LegateTask<T>::register_variant(LegateVariantCode variant_id,
                                                const VariantOptions& options)
{
  // Construct the code descriptor for this task so that the library
  // can register it later when it is ready
  Legion::CodeDescriptor code_desc(legate_task_wrapper<VARIANT_IMPL>);
  // Note that we should store the task id in a variable as the `record_variant` in the next line
  // would expect a reference of it.
  auto task_id = T::TASK_ID;
  T::Registrar::record_variant(task_id, variant_id, task_name(), VARIANT_IMPL, code_desc, options);
}

template <typename T>
template <VariantImpl VARIANT_IMPL>
/*static*/ void LegateTask<T>::add_variant(TaskInfo* task_info,
                                           LegateVariantCode variant_id,
                                           const VariantOptions& options)
{
  // Construct the code descriptor for this task so that the library
  // can register it later when it is ready
  Legion::CodeDescriptor code_desc(legate_task_wrapper<VARIANT_IMPL>);
  task_info->add_variant(variant_id, VARIANT_IMPL, code_desc, options);
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
/*static*/ void LegateTask<T>::register_variants(
  LibraryContext& context, const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = std::make_unique<TaskInfo>(task_name());
  // Make a copy of the map of options so that we can do find-or-create on it
  auto all_options_copy = all_options;
  detail::RegisterVariant<T, detail::CPUVariant>::add_variant(task_info.get(),
                                                              all_options_copy[LEGATE_CPU_VARIANT]);
  detail::RegisterVariant<T, detail::OMPVariant>::add_variant(task_info.get(),
                                                              all_options_copy[LEGATE_OMP_VARIANT]);
  detail::RegisterVariant<T, detail::GPUVariant>::add_variant(task_info.get(),
                                                              all_options_copy[LEGATE_GPU_VARIANT]);
  context.register_task(T::TASK_ID, std::move(task_info));
}

template <typename T>
/*static*/ const std::string& LegateTask<T>::task_name()
{
  static std::string result = detail::generate_task_name(typeid(T));
  return result;
}

}  // namespace legate
