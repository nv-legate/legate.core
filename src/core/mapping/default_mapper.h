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

#include "core/mapping/mapping.h"

#pragma once

namespace legate {
namespace mapping {

class DefaultMapper : public Mapper {
 public:
  virtual ~DefaultMapper() {}

 public:
  void set_machine(const MachineQueryInterface* machine);
  TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options);
  std::vector<StoreMapping> store_mappings(const Task& task,
                                           const std::vector<StoreTarget>& options);
  virtual Scalar tunable_value(TunableID tunable_id);

 private:
  const MachineQueryInterface* machine_;
};

}  // namespace mapping
}  // namespace legate
