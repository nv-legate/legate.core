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

#include "core/mapping/default_mapper.h"

namespace legate {
namespace mapping {

// Default mapper doesn't use the machine query interface
void DefaultMapper::set_machine(const MachineQueryInterface* machine) {}

TaskTarget DefaultMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return options.front();
}

std::vector<StoreMapping> DefaultMapper::store_mappings(const Task& task,
                                                        const std::vector<StoreTarget>& options)
{
  return {};
}

Scalar DefaultMapper::tunable_value(TunableID tunable_id)
{
  LEGATE_ABORT;
  return Scalar(0);
}

}  // namespace mapping
}  // namespace legate
