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

#include "core/task/registrar.h"

#include "core/runtime/context.h"
#include "core/task/task_info.h"
#include "core/utilities/typedefs.h"

namespace legate {

void TaskRegistrar::record_variant(int64_t local_task_id,
                                   LegateVariantCode variant_id,
                                   const std::string& task_name,
                                   VariantImpl body,
                                   const Legion::CodeDescriptor& code_desc,
                                   const VariantOptions& options)
{
  TaskInfo* info{nullptr};
  auto finder = pending_task_infos_.find(local_task_id);
  if (pending_task_infos_.end() == finder) {
    auto p_info = std::make_unique<TaskInfo>(task_name);
    info        = p_info.get();
    pending_task_infos_.emplace(std::make_pair(local_task_id, std::move(p_info)));
  } else
    info = finder->second.get();
#ifdef DEBUG_LEGATE
  assert(info != nullptr);
#endif
  info->add_variant(variant_id, body, code_desc, options);
}

void TaskRegistrar::register_all_tasks(LibraryContext& context)
{
  for (auto& [local_task_id, task_info] : pending_task_infos_)
    context.register_task(local_task_id, std::move(task_info));
  pending_task_infos_.clear();
}

}  // namespace legate
