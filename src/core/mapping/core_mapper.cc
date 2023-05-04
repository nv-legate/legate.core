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

#include "mappers/null_mapper.h"

#include "env_defaults.h"
#include "legate.h"

#include "core/mapping/core_mapper.h"
#include "core/mapping/machine.h"
#include "core/mapping/operation.h"
#ifdef LEGATE_USE_CUDA
#include "core/comm/comm_nccl.h"
#endif
#include "core/task/task.h"
#include "core/utilities/linearize.h"
#include "core/utilities/typedefs.h"

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
class CoreMapper : public Legion::Mapping::NullMapper {
 public:
  CoreMapper(Legion::Mapping::MapperRuntime* runtime,
             Legion::Machine machine,
             const LibraryContext* context);

  virtual ~CoreMapper();

 public:
  const char* get_mapper_name() const override;
  Legion::Mapping::Mapper::MapperSyncModel get_mapper_sync_model() const override;
  bool request_valid_instances() const override { return false; }

 public:  // Task mapping calls
  void select_task_options(const Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           TaskOptions& output) override;
  void slice_task(const Legion::Mapping::MapperContext ctx,
                  const Legion::Task& task,
                  const SliceTaskInput& input,
                  SliceTaskOutput& output) override;
  void map_task(const Legion::Mapping::MapperContext ctx,
                const Legion::Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override;
  void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                               const Legion::Task& task,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;
  void select_steal_targets(const Legion::Mapping::MapperContext ctx,
                            const SelectStealingInput& input,
                            SelectStealingOutput& output) override;
  void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput& input,
                           SelectMappingOutput& output) override;

 public:
  void configure_context(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task& task,
                         ContextConfigOutput& output) override;
  void map_future_map_reduction(const Legion::Mapping::MapperContext ctx,
                                const FutureMapReductionInput& input,
                                FutureMapReductionOutput& output) override;
  void select_tunable_value(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SelectTunableInput& input,
                            SelectTunableOutput& output) override;

 public:
  const LibraryContext* context;
  Machine machine;

 protected:
  const uint32_t min_gpu_chunk;
  const uint32_t min_cpu_chunk;
  const uint32_t min_omp_chunk;
  const uint32_t window_size;
  const uint32_t max_pending_exceptions;
  const bool precise_exception_trace;
  const uint32_t field_reuse_frac;
  const uint32_t field_reuse_freq;
  const uint32_t max_lru_length;

 private:
  std::string mapper_name;
};

CoreMapper::CoreMapper(Legion::Mapping::MapperRuntime* rt,
                       Legion::Machine m,
                       const LibraryContext* c)
  : NullMapper(rt, m),
    context(c),
    machine(m),
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
  std::stringstream ss;
  ss << context->get_library_name() << " on Node " << machine.local_node;
  mapper_name = ss.str();
}

CoreMapper::~CoreMapper(void) {}

const char* CoreMapper::get_mapper_name(void) const { return mapper_name.c_str(); }

Legion::Mapping::Mapper::MapperSyncModel CoreMapper::get_mapper_sync_model() const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void CoreMapper::select_task_options(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     TaskOptions& output)
{
  if (task.is_index_space || task.local_function) {
    Processor proc = Processor::NO_PROC;
    if (task.tag == LEGATE_CPU_VARIANT) {
      proc = machine.cpus().front();
    } else if (task.tag == LEGATE_OMP_VARIANT) {
      proc = machine.omps().front();
    } else {
      assert(task.tag == LEGATE_GPU_VARIANT);
      proc = machine.gpus().front();
    }
    output.initial_proc = proc;
    assert(output.initial_proc.exists());
    return;
  }

  mapping::Task legate_task(&task, context, runtime, ctx);
  assert(context->valid_task_id(task.task_id));
  TaskTarget target;
  switch (task.tag) {
    case LEGATE_GPU_VARIANT: {
      target = mapping::TaskTarget::GPU;
      break;
    }
    case LEGATE_OMP_VARIANT: {
      target = mapping::TaskTarget::OMP;
      break;
    }
    default: {
#ifdef DEBUG_LEGATE
      assert(LEGATE_CPU_VARIANT == task.tag);
#endif
      target = mapping::TaskTarget::CPU;
      break;
    }
  }

  auto local_range = machine.slice(target, legate_task.machine_desc(), true /*fallback_to_global*/);
#ifdef DEBUG_LEGATE
  assert(!local_range.empty());
#endif
  output.initial_proc = local_range.first();
#ifdef DEBUG_LEGATE
  assert(output.initial_proc.exists());
#endif
}

void CoreMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  assert(context->valid_task_id(task.task_id));
  output.slices.reserve(input.domain.get_volume());

  // Control-replicated because we've already been sharded
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  mapping::Task legate_task(&task, context, runtime, ctx);
  auto local_range = machine.slice(legate_task.target(), legate_task.machine_desc());

  for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
    const Point<1> point       = itr.p;
    const uint32_t local_index = point[0];
    output.slices.push_back(TaskSlice(
      Domain(itr.p, itr.p), local_range[local_index], false /*recurse*/, false /*stealable*/));
  }
}

void CoreMapper::map_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output)
{
  assert(context->valid_task_id(task.task_id));
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
  output.chosen_variant = task.tag;
}

void CoreMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  mapping::Mappable legate_mappable(&task);
  output.chosen_functor = static_cast<Legion::ShardingID>(legate_mappable.sharding_id());
}

void CoreMapper::select_steal_targets(const Legion::Mapping::MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output)
{
  // Do nothing
}

void CoreMapper::select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void CoreMapper::configure_context(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   ContextConfigOutput& output)
{
  // Use the defaults currently
}

template <typename T>
void pack_tunable(const T value, Legion::Mapping::Mapper::SelectTunableOutput& output)
{
  T* result    = static_cast<T*>(malloc(sizeof(value)));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

void CoreMapper::map_future_map_reduction(const Legion::Mapping::MapperContext ctx,
                                          const FutureMapReductionInput& input,
                                          FutureMapReductionOutput& output)
{
  output.serdez_upper_bound = LEGATE_MAX_SIZE_SCALAR_RETURN;

  if (machine.has_gpus()) {
    // TODO: It's been reported that blindly mapping target instances of future map reductions
    // to framebuffers hurts performance. Until we find a better mapping policy, we guard
    // the current policy with a macro.
#ifdef LEGATE_MAP_FUTURE_MAP_REDUCTIONS_TO_GPU

    // If this was joining exceptions, we should put instances on a host-visible memory
    // because they need serdez
    if (input.tag == LEGATE_CORE_JOIN_EXCEPTION_TAG)
      output.destination_memories.push_back(machine.zerocopy_memory());
    else
      for (auto& pair : machine.frame_buffers()) output.destination_memories.push_back(pair.second);
#else
    output.destination_memories.push_back(machine.zerocopy_memory());
#endif
  } else if (machine.has_socket_memory())
    for (auto& pair : machine.socket_memories()) output.destination_memories.push_back(pair.second);
}

void CoreMapper::select_tunable_value(const Legion::Mapping::MapperContext ctx,
                                      const Legion::Task& task,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  switch (input.tunable_id) {
    case LEGATE_CORE_TUNABLE_TOTAL_CPUS: {
      pack_tunable<int32_t>(machine.total_cpu_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_TOTAL_GPUS: {
      pack_tunable<int32_t>(machine.total_gpu_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_TOTAL_OMPS: {
      pack_tunable<int32_t>(machine.total_omp_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_NUM_PIECES: {
      if (machine.has_gpus())       // If we have GPUs, use those
        pack_tunable<int32_t>(machine.total_gpu_count(), output);
      else if (machine.has_omps())  // Otherwise use OpenMP procs
        pack_tunable<int32_t>(machine.total_omp_count(), output);
      else                          // Otherwise use the CPUs
        pack_tunable<int32_t>(machine.total_cpu_count(), output);
      return;
    }
    case LEGATE_CORE_TUNABLE_NUM_NODES: {
      pack_tunable<int32_t>(machine.total_nodes, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO: make these profile guided
      if (machine.has_gpus())
        // Make sure we can get at least 1M elements on each GPU
        pack_tunable<int64_t>(min_gpu_chunk, output);
      else if (machine.has_omps())
        // Make sure we get at least 128K elements on each OpenMP
        pack_tunable<int64_t>(min_omp_chunk, output);
      else
        // Make sure we can get at least 8KB elements on each CPU
        pack_tunable<int64_t>(min_cpu_chunk, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM: {
      pack_tunable<bool>(machine.has_socket_memory(), output);
      return;
    }
    case LEGATE_CORE_TUNABLE_WINDOW_SIZE: {
      pack_tunable<uint32_t>(window_size, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_MAX_PENDING_EXCEPTIONS: {
      pack_tunable<uint32_t>(max_pending_exceptions, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_PRECISE_EXCEPTION_TRACE: {
      pack_tunable<bool>(precise_exception_trace, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE: {
      // Multiply this by the total number of nodes and then scale by the frac
      const uint64_t global_mem_size =
        machine.has_gpus() ? machine.total_frame_buffer_size()
                           : (machine.has_socket_memory() ? machine.total_socket_memory_size()
                                                          : machine.system_memory().capacity());
      const uint64_t field_reuse_size = global_mem_size / field_reuse_frac;
      pack_tunable<uint64_t>(field_reuse_size, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_FIELD_REUSE_FREQUENCY: {
      pack_tunable<uint32_t>(field_reuse_freq, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH: {
      pack_tunable<uint32_t>(max_lru_length, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_NCCL_NEEDS_BARRIER: {
#ifdef LEGATE_USE_CUDA
      pack_tunable<bool>(machine.has_gpus() && comm::nccl::needs_barrier(), output);
#else
      pack_tunable<bool>(false, output);
#endif
      return;
    }
  }
  // Illegal tunable variable
  LEGATE_ABORT;
}

}  // namespace mapping

void register_legate_core_mapper(Legion::Machine machine,
                                 Legion::Runtime* runtime,
                                 const LibraryContext* context)
{
  // Replace all the default mappers with our custom mapper for the Legate
  // top-level task and init task
  runtime->add_mapper(context->get_mapper_id(),
                      new mapping::CoreMapper(runtime->get_mapper_runtime(), machine, context));
}

}  // namespace legate
