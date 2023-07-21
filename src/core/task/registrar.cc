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

void TaskRegistrar::record_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  pending_task_infos_.push_back(std::make_pair(local_task_id, std::move(task_info)));
}

void TaskRegistrar::register_all_tasks(LibraryContext* context)
{
  for (auto& [local_task_id, task_info] : pending_task_infos_)
    context->register_task(local_task_id, std::move(task_info));
  pending_task_infos_.clear();
}

}  // namespace legate
