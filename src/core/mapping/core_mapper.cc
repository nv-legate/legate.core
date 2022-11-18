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

#include "legate.h"

#include "core/mapping/core_mapper.h"
#include "core/mapping/machine.h"
#include "core/mapping/operation.h"
#ifdef LEGATE_USE_CUDA
#include "core/comm/comm_nccl.h"
#endif
#include "core/task/task.h"
#include "core/utilities/linearize.h"

using LegionTask    = Legion::Task;
using LegionMachine = Legion::Machine;

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {

uint32_t extract_env(const char* env_name, const uint32_t default_value, const uint32_t test_value)
{
  const char* env_value = getenv(env_name);
  if (nullptr == env_value) {
    const char* legate_test = getenv("LEGATE_TEST");
    if (legate_test != nullptr)
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
  CoreMapper(MapperRuntime* runtime, LegionMachine machine, const LibraryContext& context);
  virtual ~CoreMapper(void);

 public:
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;
  virtual bool request_valid_instances(void) const { return false; }

 public:  // Task mapping calls
  virtual void select_task_options(const MapperContext ctx,
                                   const LegionTask& task,
                                   TaskOptions& output);
  virtual void slice_task(const MapperContext ctx,
                          const LegionTask& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const LegionTask& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const LegionTask& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);
  virtual void select_steal_targets(const MapperContext ctx,
                                    const SelectStealingInput& input,
                                    SelectStealingOutput& output);
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput& input,
                                   SelectMappingOutput& output);

 public:
  virtual void configure_context(const MapperContext ctx,
                                 const LegionTask& task,
                                 ContextConfigOutput& output);
  void map_future_map_reduction(const MapperContext ctx,
                                const FutureMapReductionInput& input,
                                FutureMapReductionOutput& output);
  virtual void select_tunable_value(const MapperContext ctx,
                                    const LegionTask& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);

 protected:
  LibraryContext context;
  std::unique_ptr<Machine> machine;

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

CoreMapper::CoreMapper(MapperRuntime* rt, LegionMachine m, const LibraryContext& c)
  : NullMapper(rt, m),
    context(c),
    machine(std::make_unique<Machine>(m)),
    min_gpu_chunk(extract_env("LEGATE_MIN_GPU_CHUNK", 1 << 20, 2)),
    min_cpu_chunk(extract_env("LEGATE_MIN_CPU_CHUNK", 1 << 14, 2)),
    min_omp_chunk(extract_env("LEGATE_MIN_OMP_CHUNK", 1 << 17, 2)),
    window_size(extract_env("LEGATE_WINDOW_SIZE", 1, 1)),
    max_pending_exceptions(
      extract_env("LEGATE_MAX_PENDING_EXCEPTIONS",
#ifdef DEBUG_LEGATE
                  // In debug mode, the default is always block on tasks that can throw exceptions
                  1,
#else
                  64,
#endif
                  1)),
    precise_exception_trace(static_cast<bool>(extract_env("LEGATE_PRECISE_EXCEPTION_TRACE", 0, 0))),
    field_reuse_frac(extract_env("LEGATE_FIELD_REUSE_FRAC", 256, 256)),
    field_reuse_freq(extract_env("LEGATE_FIELD_REUSE_FREQ", 32, 32)),
    max_lru_length(extract_env("LEGATE_MAX_LRU_LENGTH", 5, 1))
{
  std::stringstream ss;
  ss << context.get_library_name() << " on Node " << machine->local_node;
  mapper_name = ss.str();
}

CoreMapper::~CoreMapper(void) {}

const char* CoreMapper::get_mapper_name(void) const { return mapper_name.c_str(); }

Mapper::MapperSyncModel CoreMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void CoreMapper::select_task_options(const MapperContext ctx,
                                     const LegionTask& task,
                                     TaskOptions& output)
{
  if (task.is_index_space) {
    Processor proc = Processor::NO_PROC;
    if (task.tag == LEGATE_CPU_VARIANT) {
      proc = machine->cpus().front();
    } else if (task.tag == LEGATE_OMP_VARIANT) {
      proc = machine->omps().front();
    } else {
      assert(task.tag == LEGATE_GPU_VARIANT);
      proc = machine->gpus().front();
    }
    output.initial_proc = proc;
    assert(output.initial_proc.exists());
    return;
  }

  mapping::Task legate_task(&task, context, runtime, ctx);
  auto& machine_desc = legate_task.machine_desc();

  Span<const Processor> avail_procs;
  uint32_t size;
  uint32_t offset;

  assert(context.valid_task_id(task.task_id));
  if (task.tag == LEGATE_CPU_VARIANT) {
    std::tie(avail_procs, size, offset) = machine_desc.slice(
      mapping::TaskTarget::CPU, machine->cpus(), machine->total_nodes, machine->local_node);
  } else if (task.tag == LEGATE_OMP_VARIANT) {
    std::tie(avail_procs, size, offset) = machine_desc.slice(
      mapping::TaskTarget::OMP, machine->omps(), machine->total_nodes, machine->local_node);
  } else {
    assert(task.tag == LEGATE_GPU_VARIANT);
    std::tie(avail_procs, size, offset) = machine_desc.slice(
      mapping::TaskTarget::GPU, machine->gpus(), machine->total_nodes, machine->local_node);
  }
  assert(avail_procs.size() > 0);
  output.initial_proc = avail_procs[0];
  assert(output.initial_proc.exists());
}

void CoreMapper::slice_task(const MapperContext ctx,
                            const LegionTask& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  assert(context.valid_task_id(task.task_id));
  output.slices.reserve(input.domain.get_volume());

  // Control-replicated because we've already been sharded
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  machine->dispatch(task.target_proc.kind(), [&](auto kind, auto& procs) {
    mapping::Task legate_task(&task, context, runtime, ctx);
    auto& machine_desc = legate_task.machine_desc();
    Span<const Processor> avail_procs;
    uint32_t size;
    uint32_t offset;
    std::tie(avail_procs, size, offset) = machine_desc.slice(
      mapping::to_target(kind), procs, machine->total_nodes, machine->local_node);

    for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
      const Point<1> point       = itr.p;
      const uint32_t local_index = point[0] % size;
      assert(local_index - offset < avail_procs.size());
      output.slices.push_back(TaskSlice(Domain(itr.p, itr.p),
                                        avail_procs[local_index - offset],
                                        false /*recurse*/,
                                        false /*stealable*/));
    }
  });
}

void CoreMapper::map_task(const MapperContext ctx,
                          const LegionTask& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output)
{
  assert(context.valid_task_id(task.task_id));
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
  output.chosen_variant = task.tag;
}

void CoreMapper::select_sharding_functor(const MapperContext ctx,
                                         const LegionTask& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  mapping::Mappable legate_mappable(&task);
  output.chosen_functor = static_cast<ShardingID>(legate_mappable.sharding_id());
}

void CoreMapper::select_steal_targets(const MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output)
{
  // Do nothing
}

void CoreMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void CoreMapper::configure_context(const MapperContext ctx,
                                   const LegionTask& task,
                                   ContextConfigOutput& output)
{
  // Use the defaults currently
}

template <typename T>
void pack_tunable(const T value, Mapper::SelectTunableOutput& output)
{
  T* result    = static_cast<T*>(malloc(sizeof(value)));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

void CoreMapper::map_future_map_reduction(const MapperContext ctx,
                                          const FutureMapReductionInput& input,
                                          FutureMapReductionOutput& output)
{
  output.serdez_upper_bound = LEGATE_MAX_SIZE_SCALAR_RETURN;

#ifdef LEGATE_MAP_FUTURE_MAP_REDUCTIONS_TO_GPU
  // TODO: It's been reported that blindly mapping target instances of future map reductions
  // to framebuffers hurts performance. Until we find a better mapping policy, we guard
  // the current policy with a macro.

  // If this was joining exceptions, we don't want to put instances anywhere
  // other than the system memory because they need serdez
  if (input.tag == LEGATE_CORE_JOIN_EXCEPTION_TAG) return;
  if (machine->has_gpus())
    for (auto& pair : local_frame_buffers) output.destination_memories.push_back(pair.second);
  else if (machine->has_socket_memory())
    for (auto& pair : local_numa_domains) output.destination_memories.push_back(pair.second);
#endif
}

void CoreMapper::select_tunable_value(const MapperContext ctx,
                                      const LegionTask& task,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  switch (input.tunable_id) {
    case LEGATE_CORE_TUNABLE_TOTAL_CPUS: {
      pack_tunable<int32_t>(machine->total_cpu_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_TOTAL_GPUS: {
      pack_tunable<int32_t>(machine->total_gpu_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_TOTAL_OMPS: {
      pack_tunable<int32_t>(machine->total_omp_count(), output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_NUM_PIECES: {
      if (machine->has_gpus())  // If we have GPUs, use those
        pack_tunable<int32_t>(machine->total_gpu_count(), output);
      else if (machine->has_omps())  // Otherwise use OpenMP procs
        pack_tunable<int32_t>(machine->total_omp_count(), output);
      else  // Otherwise use the CPUs
        pack_tunable<int32_t>(machine->total_cpu_count(), output);
      return;
    }
    case LEGATE_CORE_TUNABLE_NUM_NODES: {
      pack_tunable<int32_t>(machine->total_nodes, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO: make these profile guided
      if (machine->has_gpus())
        // Make sure we can get at least 1M elements on each GPU
        pack_tunable<int64_t>(min_gpu_chunk, output);
      else if (machine->has_omps())
        // Make sure we get at least 128K elements on each OpenMP
        pack_tunable<int64_t>(min_omp_chunk, output);
      else
        // Make sure we can get at least 8KB elements on each CPU
        pack_tunable<int64_t>(min_cpu_chunk, output);
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
        machine->has_gpus() ? machine->total_frame_buffer_size()
                            : (machine->has_socket_memory() ? machine->total_socket_memory_size()
                                                            : machine->system_memory().capacity());
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
      pack_tunable<bool>(machine->has_gpus() && comm::nccl::needs_barrier(), output);
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

void register_legate_core_mapper(LegionMachine machine,
                                 Runtime* runtime,
                                 const LibraryContext& context)
{
  // Replace all the default mappers with our custom mapper for the Legate
  // top-level task and init task
  runtime->add_mapper(context.get_mapper_id(0),
                      new mapping::CoreMapper(runtime->get_mapper_runtime(), machine, context));
}

}  // namespace legate
