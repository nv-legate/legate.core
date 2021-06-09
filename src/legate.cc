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
#include "mapper.h"
#include "shard.h"
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

/*static*/ void core_registration_callback(Machine machine,
                                           Runtime* runtime,
                                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_tasks     = LEGATE_CORE_NUM_TASK_IDS;
  config.max_shardings = 1;
  LegateContext context(runtime, core_library_name, config);

  const TaskID initialize_task_id  = context.get_task_id(LEGATE_CORE_INITIALIZE_TASK_ID);
  const char* initialize_task_name = "Legate Core Resource Initialization";
  runtime->attach_name(
    initialize_task_id, initialize_task_name, false /*mutable*/, true /*local only*/);
  const TaskID finalize_task_id  = context.get_task_id(LEGATE_CORE_FINALIZE_TASK_ID);
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

  register_legate_core_sharding_functors(runtime, context);

  // Now we can generate a mapper ID for our library and register it with the runtime
  register_legate_core_mapper(machine, runtime, context);
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
