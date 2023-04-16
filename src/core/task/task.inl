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
/*static*/ void LegateTask<T>::register_variants(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = create_task_info(all_options);
  T::Registrar::get_registrar().record_task(T::TASK_ID, std::move(task_info));
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  LibraryContext* context, const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = create_task_info(all_options);
  context->register_task(T::TASK_ID, std::move(task_info));
}

template <typename T>
/*static*/ std::unique_ptr<TaskInfo> LegateTask<T>::create_task_info(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = std::make_unique<TaskInfo>(task_name());
  detail::VariantHelper<T, detail::CPUVariant>::record(task_info.get(), all_options);
  detail::VariantHelper<T, detail::OMPVariant>::record(task_info.get(), all_options);
  detail::VariantHelper<T, detail::GPUVariant>::record(task_info.get(), all_options);
  return std::move(task_info);
}

template <typename T>
/*static*/ const std::string& LegateTask<T>::task_name()
{
  static std::string result = detail::generate_task_name(typeid(T));
  return result;
}

}  // namespace legate
