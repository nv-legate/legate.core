/* Copyright 2021-2022 NVIDIA Corporation
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

#include "env_defaults.h"
#include "legate.h"

#include "core/mapping/core_mapper.h"
#include "core/mapping/machine.h"
#ifdef LEGATE_USE_CUDA
#include "core/comm/comm_nccl.h"
#endif

namespace legate {

uint32_t extract_env(const char* env_name, const uint32_t default_value, const uint32_t test_value)
{
  const char* env_value = getenv(env_name);
  if (nullptr == env_value) {
    const char* legate_test = getenv("LEGATE_TEST");
    if (legate_test != nullptr && atoi(legate_test) > 0)
      return test_value;
    else
      return default_value;
  } else
    return atoi(env_value);
}

namespace mapping {

// This is a custom mapper implementation that only has to map
// start-up tasks associated with the Legate core, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper : public Mapper {
 public:
  CoreMapper();

 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override;
  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  legate::Scalar tunable_value(legate::TunableID tunable_id) override;

 private:
  const Machine machine;
  const int64_t min_gpu_chunk;
  const int64_t min_cpu_chunk;
  const int64_t min_omp_chunk;
  const uint32_t window_size;
  const uint32_t max_pending_exceptions;
  const bool precise_exception_trace;
  const uint32_t field_reuse_frac;
  const uint32_t field_reuse_freq;
  const uint32_t max_lru_length;
};

CoreMapper::CoreMapper()
  : machine(),
    min_gpu_chunk(extract_env("LEGATE_MIN_GPU_CHUNK", MIN_GPU_CHUNK_DEFAULT, MIN_GPU_CHUNK_TEST)),
    min_cpu_chunk(extract_env("LEGATE_MIN_CPU_CHUNK", MIN_CPU_CHUNK_DEFAULT, MIN_CPU_CHUNK_TEST)),
    min_omp_chunk(extract_env("LEGATE_MIN_OMP_CHUNK", MIN_OMP_CHUNK_DEFAULT, MIN_OMP_CHUNK_TEST)),
    window_size(extract_env("LEGATE_WINDOW_SIZE", WINDOW_SIZE_DEFAULT, WINDOW_SIZE_TEST)),
    max_pending_exceptions(extract_env("LEGATE_MAX_PENDING_EXCEPTIONS",
                                       MAX_PENDING_EXCEPTIONS_DEFAULT,
                                       MAX_PENDING_EXCEPTIONS_TEST)),
    precise_exception_trace(static_cast<bool>(extract_env("LEGATE_PRECISE_EXCEPTION_TRACE",
                                                          PRECISE_EXCEPTION_TRACE_DEFAULT,
                                                          PRECISE_EXCEPTION_TRACE_TEST))),
    field_reuse_frac(
      extract_env("LEGATE_FIELD_REUSE_FRAC", FIELD_REUSE_FRAC_DEFAULT, FIELD_REUSE_FRAC_TEST)),
    field_reuse_freq(
      extract_env("LEGATE_FIELD_REUSE_FREQ", FIELD_REUSE_FREQ_DEFAULT, FIELD_REUSE_FREQ_TEST)),
    max_lru_length(
      extract_env("LEGATE_MAX_LRU_LENGTH", MAX_LRU_LENGTH_DEFAULT, MAX_LRU_LENGTH_TEST))
{
}

void CoreMapper::set_machine(const legate::mapping::MachineQueryInterface* m) {}

TaskTarget CoreMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return options.front();
}

std::vector<StoreMapping> CoreMapper::store_mappings(const Task& task,
                                                     const std::vector<StoreTarget>& options)
{
  return {};
}

Scalar CoreMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case LEGATE_CORE_TUNABLE_TOTAL_CPUS: {
      return Scalar(int32_t(machine.total_cpu_count()));  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_TOTAL_GPUS: {
      return Scalar(int32_t(machine.total_gpu_count()));  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_TOTAL_OMPS: {
      return Scalar(int32_t(machine.total_omp_count()));  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_NUM_NODES: {
      return Scalar(int32_t(machine.total_nodes));
    }
    case LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO: make these profile guided
      if (machine.has_gpus())
        // Make sure we can get at least 1M elements on each GPU
        return Scalar(min_gpu_chunk);
      else if (machine.has_omps())
        // Make sure we get at least 128K elements on each OpenMP
        return Scalar(min_omp_chunk);
      else
        // Make sure we can get at least 8KB elements on each CPU
        return Scalar(min_cpu_chunk);
    }
    case LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM: {
      return machine.has_socket_memory();
    }
    case LEGATE_CORE_TUNABLE_WINDOW_SIZE: {
      return window_size;
    }
    case LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS: {
      return max_pending_exceptions;
    }
    case LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE: {
      return precise_exception_trace;
    }
    case LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE: {
      // Multiply this by the total number of nodes and then scale by the frac
      const uint64_t global_mem_size =
        machine.has_gpus() ? machine.total_frame_buffer_size()
                           : (machine.has_socket_memory() ? machine.total_socket_memory_size()
                                                          : machine.system_memory().capacity());
      return global_mem_size / field_reuse_frac;
    }
    case LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY: {
      return field_reuse_freq;
    }
    case LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH: {
      return max_lru_length;
    }
    case LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER: {
#ifdef LEGATE_USE_CUDA
      return machine.has_gpus() && comm::nccl::needs_barrier();
#else
      return false;
#endif
    }
  }
  // Illegal tunable variable
  LEGATE_ABORT;
  return Scalar(0);
}

std::unique_ptr<Mapper> create_core_mapper() { return std::make_unique<CoreMapper>(); }

}  // namespace mapping
}  // namespace legate
