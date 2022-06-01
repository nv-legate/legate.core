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

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <atomic>
#include <cstdlib>

#ifndef LEGATE_USE_GASNET
#include <stdint.h>
#endif

#include "coll.h"
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

using namespace Legion;
Logger log_coll("coll");

#ifdef LEGATE_USE_GASNET

enum CollTag : int {
  BCAST_TAG     = 0,
  GATHER_TAG    = 1,
  ALLTOALL_TAG  = 2,
  ALLTOALLV_TAG = 3,
  MAX_TAG       = 10,
};

static std::vector<MPI_Comm> mpi_comms;
#else  // undef LEGATE_USE_GASNET
static std::vector<ThreadComm> thread_comms;
#endif

static std::atomic<int> current_unique_id(0);

static bool coll_inited = false;

static constexpr int const MAX_NB_COMMS = 100;

// functions start here
int collCommCreate(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
#ifdef LEGATE_USE_GASNET
  int mpi_rank, mpi_comm_size;
  int *tag_ub, flag;
  int compare_result;
  MPI_Comm comm = mpi_comms[unique_id];
  CHECK_MPI(MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_result));
  assert(compare_result = MPI_CONGRUENT);

  CHECK_MPI(MPI_Comm_get_attr(comm, MPI_TAG_UB, &tag_ub, &flag));
  assert(flag);
  assert(*tag_ub == INT_MAX);
  CHECK_MPI(MPI_Comm_rank(comm, &mpi_rank));
  CHECK_MPI(MPI_Comm_size(comm, &mpi_comm_size));
  global_comm->mpi_comm_size = mpi_comm_size;
  global_comm->mpi_rank      = mpi_rank;
  global_comm->comm          = comm;
  assert(mapping_table != nullptr);
  global_comm->mapping_table.global_rank = (int*)malloc(sizeof(int) * global_comm_size);
  global_comm->mapping_table.mpi_rank    = (int*)malloc(sizeof(int) * global_comm_size);
  memcpy(global_comm->mapping_table.mpi_rank, mapping_table, sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) { global_comm->mapping_table.global_rank[i] = i; }
#else
  assert(mapping_table == nullptr);
  global_comm->mpi_comm_size = 1;
  global_comm->mpi_rank      = 0;
  if (global_comm->global_rank == 0) {
    pthread_barrier_init((pthread_barrier_t*)&(thread_comms[global_comm->unique_id].barrier),
                         nullptr,
                         global_comm->global_comm_size);
    thread_comms[global_comm->unique_id].buffers =
      (const void**)malloc(sizeof(void*) * global_comm_size);
    thread_comms[global_comm->unique_id].displs =
      (const int**)malloc(sizeof(int*) * global_comm_size);
    for (int i = 0; i < global_comm_size; i++) {
      thread_comms[global_comm->unique_id].buffers[i] = nullptr;
      thread_comms[global_comm->unique_id].displs[i]  = nullptr;
    }
    __sync_synchronize();
    thread_comms[global_comm->unique_id].ready_flag = true;
  }
  __sync_synchronize();
  volatile ThreadComm* data = &(thread_comms[global_comm->unique_id]);
  while (data->ready_flag != true) { data = &(thread_comms[global_comm->unique_id]); }
  global_comm->comm = &(thread_comms[global_comm->unique_id]);
  collBarrierLocal(global_comm);
  assert(global_comm->comm->ready_flag == true);
  assert(global_comm->comm->buffers != nullptr);
  assert(global_comm->comm->displs != nullptr);
#endif
  if (global_comm->global_comm_size % global_comm->mpi_comm_size == 0) {
    global_comm->nb_threads = global_comm->global_comm_size / global_comm->mpi_comm_size;
  } else {
    global_comm->nb_threads = global_comm->global_comm_size / global_comm->mpi_comm_size + 1;
  }
  return CollSuccess;
}

int collCommDestroy(CollComm global_comm)
{
#ifdef LEGATE_USE_GASNET
  if (global_comm->mapping_table.global_rank != nullptr) {
    free(global_comm->mapping_table.global_rank);
    global_comm->mapping_table.global_rank = nullptr;
  }
  if (global_comm->mapping_table.mpi_rank != nullptr) {
    free(global_comm->mapping_table.mpi_rank);
    global_comm->mapping_table.mpi_rank = nullptr;
  }
#else
  collBarrierLocal(global_comm);
  if (global_comm->global_rank == 0) {
    pthread_barrier_destroy((pthread_barrier_t*)&(thread_comms[global_comm->unique_id].barrier));
    free(thread_comms[global_comm->unique_id].buffers);
    thread_comms[global_comm->unique_id].buffers = nullptr;
    free(thread_comms[global_comm->unique_id].displs);
    thread_comms[global_comm->unique_id].displs = nullptr;
    __sync_synchronize();
    thread_comms[global_comm->unique_id].ready_flag = false;
  }
  __sync_synchronize();
  volatile ThreadComm* data = &(thread_comms[global_comm->unique_id]);
  while (data->ready_flag != false) { data = &(thread_comms[global_comm->unique_id]); }
#endif
  global_comm->status = false;
  return CollSuccess;
}

int collAlltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  CollDataType type,
                  CollComm global_comm)
{
  log_coll.print("Alltoallv: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#ifdef LEGATE_USE_GASNET
  return alltoallvMPI(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
#else
  return alltoallvLocal(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
#endif
}

int collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  log_coll.print("Alltoall: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#ifdef LEGATE_USE_GASNET
  return alltoallMPI(sendbuf, recvbuf, count, type, global_comm);
#else
  return alltoallLocal(sendbuf, recvbuf, count, type, global_comm);
#endif
}

int collGather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm)
{
  log_coll.print("Gather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#ifdef LEGATE_USE_GASNET
  return gatherMPI(sendbuf, recvbuf, count, type, root, global_comm);
#else
  fprintf(stderr, "Not implemented\n");
  assert(0);
#endif
}

int collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  log_coll.print("Allgather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#ifdef LEGATE_USE_GASNET
  return allgatherMPI(sendbuf, recvbuf, count, type, global_comm);
#else
  return allgatherLocal(sendbuf, recvbuf, count, type, global_comm);
#endif
}

int collBcast(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  log_coll.print("Bcast: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#ifdef LEGATE_USE_GASNET
  return bcastMPI(buf, count, type, root, global_comm);
#else
  fprintf(stderr, "Not implemented\n");
  assert(0);
#endif
}

// called from main thread
int collInit(int argc, char* argv[])
{
  current_unique_id = 0;
#ifdef LEGATE_USE_GASNET
  int provided, init_flag = 0;
  CHECK_MPI(MPI_Initialized(&init_flag));
  if (!init_flag) {
    CHECK_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
  } else {
    log_coll.print(
      "Warning: MPI has been initialized by others, make sure MPI is initialized with "
      "MPI_THREAD_MULTIPLE");
  }
  mpi_comms.resize(MAX_NB_COMMS, MPI_COMM_NULL);
  for (int i = 0; i < MAX_NB_COMMS; i++) { CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comms[i])); }
#else
  thread_comms.resize(MAX_NB_COMMS);
  for (int i = 0; i < MAX_NB_COMMS; i++) {
    thread_comms[i].ready_flag = false;
    thread_comms[i].buffers    = nullptr;
    thread_comms[i].displs     = nullptr;
  }
#endif
  coll_inited = true;
  return CollSuccess;
}

int collFinalize(void)
{
  assert(coll_inited == true);
  coll_inited = false;
#ifdef LEGATE_USE_GASNET
  for (int i = 0; i < MAX_NB_COMMS; i++) { CHECK_MPI(MPI_Comm_free(&mpi_comms[i])); }
  mpi_comms.clear();
  return MPI_Finalize();
#else
  for (int i = 0; i < MAX_NB_COMMS; i++) { assert(thread_comms[i].ready_flag == false); }
  thread_comms.clear();
  return CollSuccess;
#endif
}

int collGetUniqueId(int* id)
{
  *id = current_unique_id;
  current_unique_id++;
  assert(current_unique_id <= MAX_NB_COMMS);
  return CollSuccess;
}

#ifdef LEGATE_USE_GASNET
MPI_Datatype collDtypeToMPIDtype(CollDataType dtype)
{
  MPI_Datatype mpi_dtype = MPI_BYTE;
  switch (dtype) {
    case CollDataType::CollInt8: mpi_dtype = MPI_INT8_T; break;
    case CollDataType::CollChar: mpi_dtype = MPI_CHAR; break;
    case CollDataType::CollUint8: mpi_dtype = MPI_UINT8_T; break;
    case CollDataType::CollInt: mpi_dtype = MPI_INT; break;
    case CollDataType::CollUint32: mpi_dtype = MPI_UINT32_T; break;
    case CollDataType::CollInt64: mpi_dtype = MPI_INT64_T; break;
    case CollDataType::CollUint64: mpi_dtype = MPI_UINT64_T; break;
    case CollDataType::CollFloat: mpi_dtype = MPI_FLOAT; break;
    case CollDataType::CollDouble: mpi_dtype = MPI_DOUBLE; break;
    default: log_coll.fatal("Unknown datatype"); assert(0);
  }
  return mpi_dtype;
}

int collGenerateAlltoallTag(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // int send_tag = ((sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALL_TAG) * 10 +
  // global_comm->unique_id; // which dst seg it sends to (in dst rank) int recv_tag = ((global_rank
  // * 10000 + recvfrom_global_rank) * 10 + ALLTOALL_TAG) * 10 + global_comm->unique_id; // idx of
  // current seg we are receving (in src/my rank)
#if 1
  int tag = (rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
#else
  // still under testing
  int tag =
    ((rank1 % global_comm->nb_threads * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG) *
      MAX_NB_COMMS +
    global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

int collGenerateAlltoallvTag(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // int send_tag = ((sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALLV_TAG) * 10 +
  // global_comm->unique_id; // which dst seg it sends to (in dst rank) int recv_tag = ((global_rank
  // * 10000 + recvfrom_global_rank) * 10 + ALLTOALLV_TAG) * 10 + global_comm->unique_id; // idx of
  // current seg we are receving (in src/my rank)
#if 1
  int tag = (rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
#else
  // still under testing
  int tag = ((rank1 % global_comm->nb_threads * 10000 + rank2) * CollTag::MAX_TAG +
             CollTag::ALLTOALLV_TAG) *
              MAX_NB_COMMS +
            global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

int collGenerateBcastTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
  assert(tag < INT_MAX && tag >= 0);
  return tag;
}

int collGenerateGatherTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

#else  // undef LEGATE_USE_GASNET
size_t collGetDtypeSize(CollDataType dtype)
{
  size_t size = 0;
  switch (dtype) {
    case CollDataType::CollInt8:
    case CollDataType::CollChar: size = sizeof(char); break;
    case CollDataType::CollUint8: size = sizeof(uint8_t); break;
    case CollDataType::CollInt: size = sizeof(int); break;
    case CollDataType::CollUint32: size = sizeof(uint32_t); break;
    case CollDataType::CollInt64: size = sizeof(int64_t); break;
    case CollDataType::CollUint64: size = sizeof(uint64_t); break;
    case CollDataType::CollFloat: size = sizeof(float); break;
    case CollDataType::CollDouble: size = sizeof(double); break;
    default: log_coll.fatal("Unknown datatype"); assert(0);
  }
  return size;
}

void collUpdateBuffer(CollComm global_comm)
{
  int global_rank                         = global_comm->global_rank;
  global_comm->comm->buffers[global_rank] = nullptr;
  global_comm->comm->displs[global_rank]  = nullptr;
}

void collBarrierLocal(CollComm global_comm)
{
  assert(coll_inited == true);
  pthread_barrier_wait((pthread_barrier_t*)&(global_comm->comm->barrier));
}
#endif

void* collAllocateInplaceBuffer(const void* recvbuf, size_t size)
{
  void* sendbuf_tmp = malloc(size);
  assert(sendbuf_tmp != nullptr);
  memcpy(sendbuf_tmp, recvbuf, size);
  return sendbuf_tmp;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate