/* Copyright 2022 NVIDIA Corporation
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

#include <tuple>

#include "core/mapping/mapping.h"
#include "core/utilities/span.h"

#include "legion.h"

namespace legate {
namespace mapping {

struct ProcessorRange {
  uint32_t per_node_count{1};
  uint32_t lo{0};
  uint32_t hi{0};

  int32_t count() const { return static_cast<int32_t>(hi) - static_cast<int32_t>(lo) + 1; }
  bool empty() const { return hi < lo; }
  std::string to_string() const;
};

struct MachineDesc {
  TaskTarget preferred_target;
  std::map<TaskTarget, ProcessorRange> processor_ranges;

  ProcessorRange processor_range() const;
  std::vector<TaskTarget> valid_targets() const;
  std::tuple<Span<Legion::Processor>, uint32_t, uint32_t> slice(
    TaskTarget target,
    std::vector<Legion::Processor>& local_procs,
    uint32_t num_nodes,
    uint32_t node_id) const;
  std::string to_string() const;
};

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info);

}  // namespace mapping
}  // namespace legate
