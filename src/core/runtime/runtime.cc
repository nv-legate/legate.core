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

#include "core/mapping/core_mapper.h"
#include "core/runtime/context.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/utilities/deserializer.h"
#include "legate.h"

namespace legate {

using namespace Legion;

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress = false;

/*static*/ bool Core::use_empty_task = false;

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

  const char* empty_task = getenv("LEGATE_EMPTY_TASK");
  if (empty_task != NULL && atoi(empty_task) > 0) use_empty_task = true;
}

static ReturnValues extract_scalar_task(const Task* task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context legion_context,
                                        Runtime* runtime)
{
  TaskContext context(task, regions, legion_context, runtime);
  auto values = task->futures[0].get_result<ReturnValues>();
  auto idx    = context.scalars()[0].value<int32_t>();
  return ReturnValues({values[idx]});
}

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

void register_legate_core_tasks(Machine machine, Runtime* runtime, const LibraryContext& context)
{
  const TaskID extract_scalar_task_id  = context.get_task_id(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID);
  const char* extract_scalar_task_name = "Legate Core Scalar Extraction";
  runtime->attach_name(
    extract_scalar_task_id, extract_scalar_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar =
      make_registrar(extract_scalar_task_id, extract_scalar_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<ReturnValues, extract_scalar_task>(registrar,
                                                                      LEGATE_CPU_VARIANT);
  }
}

/*static*/ void core_registration_callback(Machine machine,
                                           Runtime* runtime,
                                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings = LEGATE_CORE_MAX_FUNCTOR_ID;
  LibraryContext context(runtime, core_library_name, config);

  register_legate_core_tasks(machine, runtime, context);

  register_legate_core_mapper(machine, runtime, context);

  register_legate_core_projection_functors(runtime, context);

  register_legate_core_sharding_functors(runtime, context);
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
