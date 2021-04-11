/* Copyright 2021 NVIDIA Corporation
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

#include "legate.h"
#include <cstdlib>
#include "legate_c.h"
#include "mappers/null_mapper.h"
#ifdef LEGATE_USE_CUDA
#include "cudalibs.h"
#endif

namespace legate {

using namespace Legion;

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress = false;

/*static*/ void Core::parse_config(void)
{
#ifndef LEGATE_USE_CUDA
  const char* need_cuda = getenv("LEGATE_NEED_CUDA");
  if (need_cuda != NULL) {
    fprintf(stderr,
            "Legate was run with GPUs but was not built with GPU support. "
            "Please install Legate again with the \"--cuda\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_OPENMP
  const char* need_openmp = getenv("LEGATE_NEED_OPENMP");
  if (need_openmp != NULL) {
    fprintf(stderr,
            "Legate was run with OpenMP processors but was not built with "
            "OpenMP support. Please install Legate again with the \"--openmp\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_GASNET
  const char* need_gasnet = getenv("LEGATE_NEED_GASNET");
  if (need_gasnet != NULL) {
    fprintf(stderr,
            "Legate was run on multiple nodes but was not built with "
            "GASNet support. Please install Legate again with the \"--gasnet\" flag.\n");
    exit(1);
  }
#endif
  const char* progress = getenv("LEGATE_SHOW_PROGRESS");
  if (progress != NULL) show_progress = true;
}

/*static*/ LayoutConstraintID Core::get_soa_layout(void)
{
  static LayoutConstraintID layout_id = 0;
  if (layout_id > 0) return layout_id;
  LayoutConstraintRegistrar constraints;
  // This should be a normal instance
  constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
  // We want C ordering of dimensions for now with fields last
  std::vector<DimensionKind> dim_order(LEGION_MAX_DIM + 1);
  for (unsigned idx = 0; idx < LEGION_MAX_DIM; ++idx)
    dim_order[idx] = static_cast<legion_dimension_kind_t>(LEGION_MAX_DIM - 1 - idx);
  dim_order[LEGION_MAX_DIM] = LEGION_DIM_F;
  constraints.add_constraint(OrderingConstraint(dim_order, true /*contiguous*/));
  Runtime* runtime = Runtime::get_runtime();
  layout_id        = runtime->register_layout(constraints);
  return layout_id;
}

/*static*/ LayoutConstraintID Core::get_reduction_layout(ReductionOpID redop)
{
  static std::map<ReductionOpID, LayoutConstraintID> reduction_layouts;
  std::map<ReductionOpID, LayoutConstraintID>::const_iterator finder =
    reduction_layouts.find(redop);
  if (finder != reduction_layouts.end()) return finder->second;
  LayoutConstraintRegistrar constraints;
  constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_REDUCTION_SPECIALIZE, redop));
  // We want C ordering of dimensions for now with fields last
  std::vector<DimensionKind> dim_order(LEGION_MAX_DIM + 1);
  for (unsigned idx = 0; idx < LEGION_MAX_DIM; ++idx)
    dim_order[idx] = static_cast<legion_dimension_kind_t>(LEGION_MAX_DIM - 1 - idx);
  dim_order[LEGION_MAX_DIM] = LEGION_DIM_F;
  constraints.add_constraint(OrderingConstraint(dim_order, true /*contiguous*/));
  Runtime* runtime             = Runtime::get_runtime();
  LayoutConstraintID layout_id = runtime->preregister_layout(constraints);
  reduction_layouts[redop]     = layout_id;
  return layout_id;
}

/*static*/ LayoutConstraintID Core::get_virtual_layout(void)
{
  static LayoutConstraintID layout_id = 0;
  if (layout_id > 0) return layout_id;
  LayoutConstraintRegistrar constraints;
  constraints.add_constraint(SpecializedConstraint(LEGION_VIRTUAL_SPECIALIZE));
  Runtime* runtime = Runtime::get_runtime();
  layout_id        = runtime->register_layout(constraints);
  return layout_id;
}

/*static*/ LegateTypeCode Core::safe_cast_type_code(int type_code)
{
#define CASE(x)                                    \
  case x: {                                        \
    return static_cast<LegateTypeCode>(type_code); \
  }
  switch (type_code) {
    CASE(BOOL_LT)
    CASE(INT8_LT)
    CASE(INT16_LT)
    CASE(INT32_LT)
    CASE(INT64_LT)
    CASE(UINT8_LT)
    CASE(UINT16_LT)
    CASE(UINT32_LT)
    CASE(UINT64_LT)
    CASE(HALF_LT)
    CASE(FLOAT_LT)
    CASE(DOUBLE_LT)
    CASE(COMPLEX64_LT)
    CASE(COMPLEX128_LT)
    default: {
      fprintf(stderr, "Invalid type code %d\n", type_code);
      LEGATE_ABORT
    }
  }

  LEGATE_ABORT
  return MAX_TYPE_NUMBER;
}

#ifdef LEGATE_USE_CUDA
static CUDALibraries& get_cuda_libraries(Processor proc, bool check)
{
  if (proc.kind() != Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT
  }
  static std::map<Processor, CUDALibraries> cuda_libraries;
  std::map<Processor, CUDALibraries>::iterator finder = cuda_libraries.find(proc);
  if (finder == cuda_libraries.end()) {
    assert(!check);
    return cuda_libraries[proc];
  } else
    return finder->second;
}

/*static*/ cublasContext* Core::get_cublas(void)
{
  const Processor executing_processor = Processor::get_executing_processor();
  CUDALibraries& lib                  = get_cuda_libraries(executing_processor, true /*check*/);
  return lib.get_cublas();
}
#endif

using namespace Legion::Mapping;

// This is a custom mapper implementation that only has to map
// start-up tasks associated with the Legate core, no one else
// should be overriding this mapper so we burry it in here
class LegateMapper : public Legion::Mapping::NullMapper {
 public:
  LegateMapper(MapperRuntime* runtime,
               Machine machine,
               TaskID core_tasks_offset,
               ShardingID core_sharding_id);
  virtual ~LegateMapper(void);

 public:
  // Start-up methods
  static AddressSpaceID get_local_node(void);
  static size_t get_total_nodes(Machine m);
  static const char* create_name(AddressSpace node);

 public:
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;
  virtual bool request_valid_instances(void) const { return false; }

 public:  // Task mapping calls
  virtual void select_task_options(const MapperContext ctx, const Task& task, TaskOptions& output);
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
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
                                 const Task& task,
                                 ContextConfigOutput& output);
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);
  void pack_tunable(const int value, Mapper::SelectTunableOutput& output);

 public:
  const AddressSpace local_node;
  const size_t total_nodes;
  const char* const mapper_name;
  const TaskID core_tasks_offset;
  const ShardingID core_sharding_id;

 protected:
  std::vector<Processor> local_cpus;
  std::vector<Processor> local_gpus;

 protected:
  Memory local_system_memory, local_zerocopy_memory;
  std::map<Processor, Memory> local_frame_buffers;
};

LegateMapper::LegateMapper(MapperRuntime* rt, Machine m, TaskID offset, ShardingID core_shard)
  : NullMapper(rt, m),
    local_node(get_local_node()),
    total_nodes(get_total_nodes(m)),
    mapper_name(create_name(local_node)),
    core_tasks_offset(offset),
    core_sharding_id(core_shard)
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (Machine::ProcessorQuery::iterator it = local_procs.begin(); it != local_procs.end(); it++) {
    switch (it->kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(*it);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(*it);
        break;
      }
      default: break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (std::vector<Processor>::const_iterator it = local_gpus.begin(); it != local_gpus.end();
       it++) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(*it);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[*it] = local_framebuffer.first();
  }
}

LegateMapper::~LegateMapper(void) { free(const_cast<char*>(mapper_name)); }

/*static*/ AddressSpace LegateMapper::get_local_node(void)
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

/*static*/ size_t LegateMapper::get_total_nodes(Machine m)
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); it != query.end(); it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

/*static*/ const char* LegateMapper::create_name(AddressSpace node)
{
  char buffer[128];
  snprintf(buffer, 127, "Legate Mapper on Node %d", node);
  return strdup(buffer);
}

const char* LegateMapper::get_mapper_name(void) const { return mapper_name; }

Mapper::MapperSyncModel LegateMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void LegateMapper::select_task_options(const MapperContext ctx,
                                       const Task& task,
                                       TaskOptions& output)
{
  assert(core_tasks_offset <= task.task_id);
  assert(task.task_id < (core_tasks_offset + LEGATE_CORE_NUM_TASK_IDS));
  if (task.tag == LEGATE_CPU_VARIANT) {
    assert(!local_cpus.empty());
    output.initial_proc = local_cpus.front();
  } else {
    assert(task.tag == LEGATE_GPU_VARIANT);
    assert(!local_gpus.empty());
    output.initial_proc = local_gpus.front();
  }
}

void LegateMapper::slice_task(const MapperContext ctx,
                              const Task& task,
                              const SliceTaskInput& input,
                              SliceTaskOutput& output)
{
  assert(core_tasks_offset <= task.task_id);
  assert(task.task_id < (core_tasks_offset + LEGATE_CORE_NUM_TASK_IDS));
  output.slices.reserve(input.domain.get_volume());
  // Check to see if we're control replicated or not. If we are then
  // we'll already have been sharded.
  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(task.target_proc.kind());
  if (all_procs.count() == input.domain.get_volume()) {
    Machine::ProcessorQuery::iterator pit = all_procs.begin();
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++, pit++)
      output.slices.push_back(
        TaskSlice(Domain(itr.p, itr.p), *pit, false /*recurse*/, false /*stealable*/));
  } else {
    // Control-replicated because we've already been sharded
    Domain sharding_domain = task.index_domain;
    if (task.sharding_space.exists())
      sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);
    assert(sharding_domain.get_dim() == 1);
    assert(input.domain.get_dim() == 1);
    const Rect<1> space = sharding_domain;
    const Rect<1> local = input.domain;
    const size_t size   = (space.hi[0] - space.lo[0]) + 1;
    // Assume that if we're control replicated there is one shard per space
    const coord_t chunk = (size + total_nodes - 1) / total_nodes;
    const coord_t start = local_node * chunk + space.lo[0];
    switch (task.target_proc.kind()) {
      case Processor::LOC_PROC: {
        for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
          const Point<1> point = itr.p;
          assert(point[0] >= start);
          assert(point[0] < (start + chunk));
          const unsigned local_index = point[0] - start;
          assert(local_index < local_cpus.size());
          output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), local_cpus[local_index], false /*recurse*/, false /*stealable*/));
        }
        break;
      }
      case Processor::TOC_PROC: {
        for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
          const Point<1> point = itr.p;
          assert(point[0] >= start);
          assert(point[0] < (start + chunk));
          const unsigned local_index = point[0] - start;
          assert(local_index < local_gpus.size());
          output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), local_gpus[local_index], false /*recurse*/, false /*stealable*/));
        }
        break;
      }
      default: LEGATE_ABORT
    }
  }
}

void LegateMapper::map_task(const MapperContext ctx,
                            const Task& task,
                            const MapTaskInput& input,
                            MapTaskOutput& output)
{
  assert(core_tasks_offset <= task.task_id);
  assert(task.task_id < (core_tasks_offset + LEGATE_CORE_NUM_TASK_IDS));
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
  output.chosen_variant = task.tag;
}

void LegateMapper::select_sharding_functor(const MapperContext ctx,
                                           const Task& task,
                                           const SelectShardingFunctorInput& input,
                                           SelectShardingFunctorOutput& output)
{
  assert(core_tasks_offset <= task.task_id);
  assert(task.task_id < (core_tasks_offset + LEGATE_CORE_NUM_TASK_IDS));
  assert(task.regions.empty());
  const int launch_dim = task.index_domain.get_dim();
  assert(launch_dim == 1);
  output.chosen_functor = core_sharding_id;
}

void LegateMapper::select_steal_targets(const MapperContext ctx,
                                        const SelectStealingInput& input,
                                        SelectStealingOutput& output)
{
  // Do nothing
}

void LegateMapper::select_tasks_to_map(const MapperContext ctx,
                                       const SelectMappingInput& input,
                                       SelectMappingOutput& output)
{
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void LegateMapper::configure_context(const MapperContext ctx,
                                     const Task& task,
                                     ContextConfigOutput& output)
{
  // Use the defaults currently
}

void LegateMapper::pack_tunable(const int value, Mapper::SelectTunableOutput& output)
{
  int* result  = (int*)malloc(sizeof(value));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

void LegateMapper::select_tunable_value(const MapperContext ctx,
                                        const Task& task,
                                        const SelectTunableInput& input,
                                        SelectTunableOutput& output)
{
  if (input.tunable_id == LEGATE_CORE_TUNABLE_TOTAL_CPUS) {
    pack_tunable(local_cpus.size() * total_nodes, output);  // assume symmetry
  } else if (input.tunable_id == LEGATE_CORE_TUNABLE_TOTAL_GPUS) {
    pack_tunable(local_gpus.size() * total_nodes, output);  // assume symmetry
  } else {                                                  // Illegaal tunable variable
    LEGATE_ABORT
  }
}

static void initialize_cpu_resource_task(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Runtime* runtime)
{
  // Nothing to do here yet...
}

static void finalize_cpu_resource_task(const Task* task,
                                       const std::vector<PhysicalRegion>& regions,
                                       Context ctx,
                                       Runtime* runtime)
{
  // Nothing to do here yet...
}

#ifdef LEGATE_USE_CUDA
static void initialize_gpu_resource_task(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Runtime* runtime)
{
  const LegateResource resource = *((const LegateResource*)task->args);
  switch (resource) {
    case LEGATE_CORE_RESOURCE_CUBLAS: {
      // This call will initialize cublas
      Core::get_cublas();
      break;
    }
    // TODO: implement support for other libraries
    case LEGATE_CORE_RESOURCE_CUDNN:
    case LEGATE_CORE_RESOURCE_CUDF:
    case LEGATE_CORE_RESOURCE_CUML:
    default: LEGATE_ABORT
  }
}

static void finalize_gpu_resource_task(const Task* task,
                                       const std::vector<PhysicalRegion>& regions,
                                       Context ctx,
                                       Runtime* runtime)
{
  CUDALibraries& libs = get_cuda_libraries(task->current_proc, true /*check*/);
  libs.finalize();
}
#endif  // LEGATE_USE_CUDA

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

class LegateCoreShardingFunctor : public ShardingFunctor {
 public:
  virtual ShardID shard(const DomainPoint& p, const Domain& launch_space, const size_t total_shards)
  {
    // Just tile this space in 1D
    const Point<1> point = p;
    const Rect<1> space  = launch_space;
    const size_t size    = (space.hi[0] - space.lo[0]) + 1;
    const size_t chunk   = (size + total_shards - 1) / total_shards;
    return (point[0] - space.lo[0]) / chunk;
  }
};

/*static*/ void core_registration_callback(Machine machine,
                                           Runtime* runtime,
                                           const std::set<Processor>& local_procs)
{
  const TaskID core_tasks_offset =
    runtime->generate_library_task_ids(core_library_name, LEGATE_CORE_NUM_TASK_IDS);
  const TaskID initialize_task_id  = core_tasks_offset + LEGATE_CORE_INITIALIZE_TASK_ID;
  const char* initialize_task_name = "Legate Core Resource Initialization";
  runtime->attach_name(
    initialize_task_id, initialize_task_name, false /*mutable*/, true /*local only*/);
  const TaskID finalize_task_id  = core_tasks_offset + LEGATE_CORE_FINALIZE_TASK_ID;
  const char* finalize_task_name = "Legate Core Resource Finalization";
  runtime->attach_name(
    finalize_task_id, finalize_task_name, false /*mutable*/, true /*local only*/);
  // Register the task variant for both CPUs and GPUs
  {
    TaskVariantRegistrar registrar(initialize_task_id, initialize_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    runtime->register_task_variant<initialize_cpu_resource_task>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    TaskVariantRegistrar registrar(finalize_task_id, finalize_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    runtime->register_task_variant<finalize_cpu_resource_task>(registrar, LEGATE_CPU_VARIANT);
  }
#ifdef LEGATE_USE_CUDA
  {
    TaskVariantRegistrar registrar(initialize_task_id, initialize_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    runtime->register_task_variant<initialize_gpu_resource_task>(registrar, LEGATE_GPU_VARIANT);
    // Make sure we fill in all the cuda libraries entries for the
    // local processors so we don't have races later
    Machine::ProcessorQuery local_gpus(machine);
    local_gpus.local_address_space();
    local_gpus.only_kind(Processor::TOC_PROC);
    for (Machine::ProcessorQuery::iterator it = local_gpus.begin(); it != local_gpus.end(); it++) {
      // This call will make an entry for the CUDA libraries but not
      // initialize any of them
      get_cuda_libraries(*it, false /*check*/);
    }
  }
  {
    TaskVariantRegistrar registrar(finalize_task_id, finalize_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    runtime->register_task_variant<finalize_gpu_resource_task>(registrar, LEGATE_GPU_VARIANT);
  }
#endif

  // Generate a library sharding ID in case we end up being control replicated
  const ShardingID sharding_id = runtime->generate_library_sharding_ids(core_library_name, 1);
  runtime->register_sharding_functor(sharding_id, new LegateCoreShardingFunctor());

  // Now we can generate a mapper ID for our library and register it with the runtime
  const MapperID core_mapper_id = runtime->generate_library_mapper_ids(core_library_name, 1);
  // Replace all the default mappers with our custom mapper for the Legate
  // top-level task and init task
  runtime->add_mapper(
    core_mapper_id,
    new LegateMapper(runtime->get_mapper_runtime(), machine, core_tasks_offset, sharding_id));
}

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::core_registration_callback,
                                                 true /*global*/);
}
}
