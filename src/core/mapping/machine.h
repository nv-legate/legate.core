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

TaskTarget to_target(Processor::Kind kind);

Processor::Kind to_kind(TaskTarget target);

struct ProcessorRange {
  ProcessorRange() {}
  ProcessorRange(uint32_t low, uint32_t high, uint32_t per_node_count);
  ProcessorRange operator&(const ProcessorRange&) const;
  uint32_t count() const;
  bool empty() const;
  std::string to_string() const;

  uint32_t low{0};
  uint32_t high{0};
  uint32_t per_node_count{1};
};

struct MachineDesc {
  TaskTarget preferred_target;
  std::map<TaskTarget, ProcessorRange> processor_ranges;

  ProcessorRange processor_range() const;
  std::vector<TaskTarget> valid_targets() const;
  std::vector<TaskTarget> valid_targets_except(std::set<TaskTarget>&& to_exclude) const;
  std::string to_string() const;
};

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info);

class Machine;

class LocalProcessorRange {
 private:
  friend class Machine;
  LocalProcessorRange();
  LocalProcessorRange(const std::vector<Processor>& procs);
  LocalProcessorRange(uint32_t offset,
                      uint32_t total_proc_count,
                      const Processor* local_procs,
                      size_t num_local_procs);

 public:
  const Processor& first() const { return *procs_.begin(); }
  const Processor& operator[](uint32_t idx) const;

 public:
  bool empty() const { return procs_.size() == 0; }

 private:
  uint32_t offset_;
  uint32_t total_proc_count_;
  Span<const Processor> procs_;
};

class Machine {
 public:
  Machine();

 public:
  const std::vector<Processor>& cpus() const { return cpus_; }
  const std::vector<Processor>& gpus() const { return gpus_; }
  const std::vector<Processor>& omps() const { return omps_; }
  const std::vector<Processor>& procs(TaskTarget target) const;

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
  Memory get_memory(Processor proc, StoreTarget target) const;
  Memory system_memory() const { return system_memory_; }
  Memory zerocopy_memory() const { return zerocopy_memory_; }
  const std::map<Processor, Memory>& frame_buffers() const { return frame_buffers_; }
  const std::map<Processor, Memory>& socket_memories() const { return socket_memories_; }

 public:
  LocalProcessorRange slice(TaskTarget target,
                            const MachineDesc& machine_desc,
                            bool fallback_to_global = false) const;

 public:
  const uint32_t local_node;
  const uint32_t total_nodes;

 private:
  std::vector<Processor> cpus_;
  std::vector<Processor> gpus_;
  std::vector<Processor> omps_;

 private:
  Memory system_memory_, zerocopy_memory_;
  std::map<Processor, Memory> frame_buffers_;
  std::map<Processor, Memory> socket_memories_;
};

}  // namespace mapping
}  // namespace legate
