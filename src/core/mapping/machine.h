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
#include "legate_defines.h"
#include "legion.h"

namespace legate {
namespace mapping {

TaskTarget to_target(Legion::Processor::Kind kind);

Legion::Processor::Kind to_kind(TaskTarget target);

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
  std::vector<TaskTarget> valid_targets(std::set<TaskTarget>&& to_exclude) const;
  std::tuple<Span<const Legion::Processor>, uint32_t, uint32_t> slice(
    TaskTarget target,
    const std::vector<Legion::Processor>& local_procs,
    uint32_t num_nodes,
    uint32_t node_id) const;
  std::string to_string() const;
};

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info);

class Machine {
 public:
  Machine(Legion::Machine legion_machine);

 public:
  const std::vector<Legion::Processor>& cpus() const { return cpus_; }
  const std::vector<Legion::Processor>& gpus() const { return gpus_; }
  const std::vector<Legion::Processor>& omps() const { return omps_; }

 public:
  size_t total_cpu_count() const { return total_nodes * cpus_.size(); }
  size_t total_gpu_count() const { return total_nodes * gpus_.size(); }
  size_t total_omp_count() const { return total_nodes * omps_.size(); }

 public:
  size_t total_frame_buffer_size() const;
  size_t total_socket_memory_size() const;

 public:
  bool has_cpus() const { return !cpus_.empty(); }
  bool has_gpus() const { return !gpus_.empty(); }
  bool has_omps() const { return !omps_.empty(); }

 public:
  bool has_socket_memory() const;

 public:
  Legion::Memory get_memory(Legion::Processor proc, StoreTarget target) const;
  Legion::Memory system_memory() const { return system_memory_; }
  Legion::Memory zerocopy_memory() const { return zerocopy_memory_; }

 public:
  template <typename Functor>
  decltype(auto) dispatch(TaskTarget target, Functor functor) const
  {
    switch (target) {
      case TaskTarget::CPU: return functor(target, cpus_);
      case TaskTarget::GPU: return functor(target, gpus_);
      case TaskTarget::OMP: return functor(target, omps_);
    }
    assert(false);
    return functor(target, cpus_);
  }
  template <typename Functor>
  decltype(auto) dispatch(Legion::Processor::Kind kind, Functor functor) const
  {
    switch (kind) {
      case Legion::Processor::LOC_PROC: return functor(kind, cpus_);
      case Legion::Processor::TOC_PROC: return functor(kind, gpus_);
      case Legion::Processor::OMP_PROC: return functor(kind, omps_);
      default: LEGATE_ABORT;
    }
    assert(false);
    return functor(kind, cpus_);
  }

 public:
  const uint32_t local_node;
  const uint32_t total_nodes;

 private:
  std::vector<Legion::Processor> cpus_;
  std::vector<Legion::Processor> gpus_;
  std::vector<Legion::Processor> omps_;

 private:
  Legion::Memory system_memory_, zerocopy_memory_;
  std::map<Legion::Processor, Legion::Memory> frame_buffers_;
  std::map<Legion::Processor, Legion::Memory> socket_memories_;
};

}  // namespace mapping
}  // namespace legate
