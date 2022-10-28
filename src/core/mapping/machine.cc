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

#include "core/mapping/machine.h"

namespace legate {
namespace mapping {

using namespace Legion;

std::ostream& operator<<(std::ostream& stream, const TaskTarget& target)
{
  switch (target) {
    case TaskTarget::GPU: {
      stream << "GPU";
      break;
    }
    case TaskTarget::OMP: {
      stream << "OMP";
      break;
    }
    case TaskTarget::CPU: {
      stream << "CPU";
      break;
    }
  }
  return stream;
}

std::pair<uint32_t, uint32_t> MachineDesc::processor_range() const
{
  auto finder = processor_ranges.find(preferred_target);
#ifdef DEBUG_LEGATE
  assert(finder != processor_ranges.end());
#endif
  return finder->second;
}

std::vector<TaskTarget> MachineDesc::valid_targets() const
{
  std::vector<TaskTarget> result;
  for (auto& [target, _] : processor_ranges) result.push_back(target);
  return std::move(result);
}

std::tuple<Span<Processor>, uint32_t, uint32_t> MachineDesc::slice(
  TaskTarget target,
  std::vector<Processor>& local_procs,
  uint32_t num_nodes,
  uint32_t node_id) const
{
  auto finder = processor_ranges.find(target);
  if (processor_ranges.end() == finder)
    return std::make_tuple(Span<Processor>(nullptr, 0), uint32_t(1), uint32_t(0));

  auto& range = finder->second;

  // TODO: Let's assume nodes are homogeneous for now

  uint32_t num_procs = local_procs.size();
  uint32_t global_lo = num_procs * node_id;
  uint32_t global_hi = global_lo + num_procs - 1;

  uint32_t my_lo = std::max(range.first, global_lo) - global_lo;
  uint32_t my_hi = std::min(range.second, global_hi) - global_lo;

  uint32_t size   = range.second - range.first + 1;
  uint32_t offset = (my_lo + global_lo) - range.first;

  return std::make_tuple(
    Span<Processor>(local_procs.data() + my_lo, my_hi - my_lo + 1), size, offset);
}

std::string MachineDesc::to_string() const
{
  std::stringstream ss;
  ss << "Machine(preferred_kind: " << preferred_target;
  for (auto& [kind, range] : processor_ranges)
    ss << ", " << kind << ": [" << range.first << ", " << range.second << "]";
  ss << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info)
{
  stream << info.to_string();
  return stream;
}

}  // namespace mapping
}  // namespace legate
