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

#include "core/comm/comm_nccl.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/data/buffer.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"
#include "legate.h"

#include <cuda.h>
#include <nccl.h>

namespace legate {
namespace comm {
namespace nccl {

struct _Payload {
  uint64_t field0;
  uint64_t field1;
};

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
    exit(error);
  }
}

static ncclUniqueId init_nccl_id(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& regions,
                                 Legion::Context context,
                                 Legion::Runtime* runtime)
{
  legate::nvtx::Range auto_range("core::comm::nccl::init_id");

  Core::show_progress(task, context, runtime);

  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  return id;
}

static ncclComm_t* init_nccl(const Legion::Task* task,
                             const std::vector<Legion::PhysicalRegion>& regions,
                             Legion::Context context,
                             Legion::Runtime* runtime)
{
  legate::nvtx::Range auto_range("core::comm::nccl::init");

  Core::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);

  auto id          = task->futures[0].get_result<ncclUniqueId>();
  ncclComm_t* comm = new ncclComm_t{};
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclCommInitRank(comm, task->index_domain.get_volume(), id, task->index_point[0]));
  CHECK_NCCL(ncclGroupEnd());

  auto num_ranks = task->index_domain.get_volume();

  if (num_ranks == 1) return comm;

  auto stream = cuda::StreamPool::get_stream_pool().get_stream();

  // Perform a warm-up all-to-all

  auto src_buffer = create_buffer<_Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto tgt_buffer = create_buffer<_Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);

  CHECK_NCCL(ncclGroupStart());
  for (auto idx = 0; idx < num_ranks; ++idx) {
    CHECK_NCCL(ncclSend(src_buffer.ptr(0), sizeof(_Payload), ncclInt8, idx, *comm, stream));
    CHECK_NCCL(ncclRecv(tgt_buffer.ptr(0), sizeof(_Payload), ncclInt8, idx, *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  CHECK_NCCL(ncclAllGather(src_buffer.ptr(0), tgt_buffer.ptr(0), 1, ncclUint64, *comm, stream));

  return comm;
}

static void finalize_nccl(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context context,
                          Legion::Runtime* runtime)
{
  legate::nvtx::Range auto_range("core::comm::nccl::finalize");

  Core::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);
  auto comm = task->futures[0].get_result<ncclComm_t*>();
  CHECK_NCCL(ncclCommDestroy(*comm));
  delete comm;
}

void register_tasks(Legion::Machine machine,
                    Legion::Runtime* runtime,
                    const LibraryContext* context)
{
  auto init_nccl_id_task_id          = context->get_task_id(LEGATE_CORE_INIT_NCCL_ID_TASK_ID);
  const char* init_nccl_id_task_name = "core::comm::nccl::init_id";
  runtime->attach_name(
    init_nccl_id_task_id, init_nccl_id_task_name, false /*mutable*/, true /*local only*/);

  auto init_nccl_task_id          = context->get_task_id(LEGATE_CORE_INIT_NCCL_TASK_ID);
  const char* init_nccl_task_name = "core::comm::nccl::init";
  runtime->attach_name(
    init_nccl_task_id, init_nccl_task_name, false /*mutable*/, true /*local only*/);

  auto finalize_nccl_task_id          = context->get_task_id(LEGATE_CORE_FINALIZE_NCCL_TASK_ID);
  const char* finalize_nccl_task_name = "core::comm::nccl::finalize";
  runtime->attach_name(
    finalize_nccl_task_id, finalize_nccl_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar =
      make_registrar(init_nccl_id_task_id, init_nccl_id_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<ncclUniqueId, init_nccl_id>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar = make_registrar(init_nccl_task_id, init_nccl_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<ncclComm_t*, init_nccl>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_nccl_task_id, finalize_nccl_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<finalize_nccl>(registrar, LEGATE_GPU_VARIANT);
  }
}

bool needs_barrier()
{
  // Blocking communications in NCCL violate CUDA's (undocumented) concurrent forward progress
  // requirements and no CUDA drivers that have released are safe from this. Until either CUDA
  // or NCCL is fixed, we will always insert a barrier at the beginning of every NCCL task.
  return true;
}

}  // namespace nccl
}  // namespace comm
}  // namespace legate
