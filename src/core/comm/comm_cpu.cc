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

#include "core/comm/comm_cpu.h"
#include "legate.h"

#include "core/comm/coll.h"

using namespace Legion;

namespace legate {
namespace comm {
namespace cpu {

static int init_coll_cpu_mapping(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& regions,
                                 Legion::Context context,
                                 Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime, task->get_task_name());
  int mpi_rank = 0;
#if defined (LEGATE_USE_GASNET)
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  return mpi_rank;
}

static collComm_t init_coll_cpu(const Legion::Task* task,
                                const std::vector<Legion::PhysicalRegion>& regions,
                                Legion::Context context,
                                Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime, task->get_task_name());

  const int point = task->index_point[0];
  int num_ranks = task->index_domain.get_volume();
  assert(task->futures.size() == static_cast<size_t>(num_ranks));
  int *mapping_table = (int *)malloc(sizeof(int) * num_ranks);
  for (int i = 0; i < num_ranks; i++) {
    const int* mapping_table_element = (const int*)task->futures[i].get_buffer(Memory::SYSTEM_MEM);
    mapping_table[i] = *mapping_table_element;
  }

  collComm_t comm = (collComm_t)malloc(sizeof(Coll_Comm));

 #if defined (LEGATE_USE_GASNET)
  collCommCreate(comm, num_ranks, point, mapping_table);
#else
  collCommCreate(comm, num_ranks, point, NULL);
#endif

  assert(mapping_table[point] == comm->mpi_rank);
  free(mapping_table);
  return comm;
}

static void finalize_coll_cpu(const Legion::Task* task,
                              const std::vector<Legion::PhysicalRegion>& regions,
                              Legion::Context context,
                              Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime, task->get_task_name());

  assert(task->futures.size() == 1);
  collComm_t comm = task->futures[0].get_result<collComm_t>();
  const int point = task->index_point[0];
  assert(comm->global_rank == point);
  collCommDestroy(comm);
  free(comm);
  comm = NULL;
}

void register_tasks(Legion::Machine machine,
                    Legion::Runtime* runtime,
                    const LibraryContext& context)
{
  const InputArgs &command_args = Legion::Runtime::get_input_args();
  int argc = command_args.argc;
  char **argv = command_args.argv;
  collInit(argc, argv, 0);

  const TaskID init_coll_cpu_mapping_task_id  = context.get_task_id(LEGATE_CORE_INIT_COLL_CPU_MAPPING_TASK_ID);
  const char* init_coll_cpu_mapping_task_name = "core::comm::cpu::init_mapping";
  runtime->attach_name(
    init_coll_cpu_mapping_task_id, init_coll_cpu_mapping_task_name, false /*mutable*/, true /*local only*/);

  const TaskID init_coll_cpu_task_id  = context.get_task_id(LEGATE_CORE_INIT_COLL_CPU_TASK_ID);
  const char* init_coll_cpu_task_name = "core::comm::cpu::init";
  runtime->attach_name(
    init_coll_cpu_task_id, init_coll_cpu_task_name, false /*mutable*/, true /*local only*/);

  const TaskID finalize_coll_cpu_task_id  = context.get_task_id(LEGATE_CORE_FINALIZE_COLL_CPU_TASK_ID);
  const char* finalize_coll_cpu_task_name = "core::comm::cpu::finalize";
  runtime->attach_name(
    finalize_coll_cpu_task_id, finalize_coll_cpu_task_name, false /*mutable*/, true /*local only*/);

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
      make_registrar(init_coll_cpu_mapping_task_id, init_coll_cpu_mapping_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<int, init_coll_cpu_mapping>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar = make_registrar(init_coll_cpu_task_id, init_coll_cpu_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<collComm_t, init_coll_cpu>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_coll_cpu_task_id, finalize_coll_cpu_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<finalize_coll_cpu>(registrar, LEGATE_CPU_VARIANT);
  }
}

}  // namespace cpu
}  // namespace comm
}  // namespace legate
