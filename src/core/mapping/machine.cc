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

TaskTarget to_target(Legion::Processor::Kind kind)
{
  switch (kind) {
    case Legion::Processor::Kind::TOC_PROC: return TaskTarget::GPU;
    case Legion::Processor::Kind::OMP_PROC: return TaskTarget::OMP;
    case Legion::Processor::Kind::LOC_PROC: return TaskTarget::CPU;
    default: LEGATE_ABORT;
  }
  assert(false);
  return TaskTarget::CPU;
}

Legion::Processor::Kind to_kind(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return Legion::Processor::Kind::TOC_PROC;
    case TaskTarget::OMP: return Legion::Processor::Kind::OMP_PROC;
    case TaskTarget::CPU: return Legion::Processor::Kind::LOC_PROC;
    default: LEGATE_ABORT;
  }
  assert(false);
  return Legion::Processor::Kind::LOC_PROC;
}

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

std::string ProcessorRange::to_string() const
{
  std::stringstream ss;
  ss << "Proc([" << lo << "," << hi << "], " << per_node_count << " per node)";
  return ss.str();
}

ProcessorRange MachineDesc::processor_range() const
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

std::vector<TaskTarget> MachineDesc::valid_targets(std::set<TaskTarget>&& to_exclude) const
{
  std::vector<TaskTarget> result;
  for (auto& [target, _] : processor_ranges)
    if (to_exclude.find(target) == to_exclude.end()) result.push_back(target);
  return std::move(result);
}

std::tuple<Span<const Legion::Processor>, uint32_t, uint32_t> MachineDesc::slice(
  TaskTarget target,
  const std::vector<Legion::Processor>& local_procs,
  uint32_t num_nodes,
  uint32_t node_id) const
{
  auto finder = processor_ranges.find(target);
  if (processor_ranges.end() == finder)
    return std::make_tuple(Span<const Legion::Processor>(nullptr, 0), uint32_t(1), uint32_t(0));

  auto& range = finder->second;

  // TODO: Let's assume nodes are homogeneous for now

  uint32_t num_procs = local_procs.size();
  uint32_t global_lo = num_procs * node_id;
  uint32_t global_hi = global_lo + num_procs - 1;

  uint32_t my_lo = std::max(range.lo, global_lo) - global_lo;
  uint32_t my_hi = std::min(range.hi, global_hi) - global_lo;

  uint32_t size   = range.hi - range.lo + 1;
  uint32_t offset = (my_lo + global_lo) - range.lo;

  return std::make_tuple(
    Span<const Legion::Processor>(local_procs.data() + my_lo, my_hi - my_lo + 1), size, offset);
}

std::string MachineDesc::to_string() const
{
  std::stringstream ss;
  ss << "Machine(preferred_kind: " << preferred_target;
  for (auto& [kind, range] : processor_ranges) ss << ", " << kind << ": " << range.to_string();
  ss << ")";
  return ss.str();
}

Machine::Machine(Legion::Machine legion_machine)
  : local_node(Legion::Processor::get_executing_processor().address_space()),
    total_nodes(legion_machine.get_address_space_count())
{
  Legion::Machine::ProcessorQuery procs(legion_machine);
  // Query to find all our local processors
  procs.local_address_space();
  for (auto proc : procs) {
    switch (proc.kind()) {
      case Legion::Processor::LOC_PROC: {
        cpus_.push_back(proc);
        continue;
      }
      case Legion::Processor::TOC_PROC: {
        gpus_.push_back(proc);
        continue;
      }
      case Legion::Processor::OMP_PROC: {
        omps_.push_back(proc);
        continue;
      }
    }
  }

  // Now do queries to find all our local memories
  Legion::Machine::MemoryQuery sysmem(legion_machine);
  sysmem.local_address_space().only_kind(Legion::Memory::SYSTEM_MEM);
  assert(sysmem.count() > 0);
  system_memory_ = sysmem.first();

  if (!gpus_.empty()) {
    Legion::Machine::MemoryQuery zcmem(legion_machine);
    zcmem.local_address_space().only_kind(Legion::Memory::Z_COPY_MEM);
    assert(zcmem.count() > 0);
    zerocopy_memory_ = zcmem.first();
  }
  for (auto& gpu : gpus_) {
    Legion::Machine::MemoryQuery framebuffer(legion_machine);
    framebuffer.local_address_space().only_kind(Legion::Memory::GPU_FB_MEM).best_affinity_to(gpu);
    assert(framebuffer.count() > 0);
    frame_buffers_[gpu] = framebuffer.first();
  }
  for (auto& omp : omps_) {
    Legion::Machine::MemoryQuery sockmem(legion_machine);
    sockmem.local_address_space().only_kind(Legion::Memory::SOCKET_MEM).best_affinity_to(omp);
    // If we have socket memories then use them
    if (sockmem.count() > 0) socket_memories_[omp] = sockmem.first();
    // Otherwise we just use the local system memory
    else
      socket_memories_[omp] = system_memory_;
  }
}

size_t Machine::total_frame_buffer_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  size_t per_node_size = frame_buffers_.size() * frame_buffers_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

size_t Machine::total_socket_memory_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  size_t per_node_size = socket_memories_.size() * socket_memories_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

bool Machine::has_socket_memory() const
{
  return !socket_memories_.empty() &&
         socket_memories_.begin()->second.kind() == Legion::Memory::SOCKET_MEM;
}

Legion::Memory Machine::get_memory(Legion::Processor proc, StoreTarget target) const
{
  switch (target) {
    case StoreTarget::SYSMEM: return system_memory_;
    case StoreTarget::FBMEM: return frame_buffers_.at(proc);
    case StoreTarget::ZCMEM: return zerocopy_memory_;
    case StoreTarget::SOCKETMEM: return socket_memories_.at(proc);
    default: LEGATE_ABORT;
  }
  assert(false);
  return Legion::Memory::NO_MEMORY;
}

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info)
{
  stream << info.to_string();
  return stream;
}

}  // namespace mapping
}  // namespace legate
