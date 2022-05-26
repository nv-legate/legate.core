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

#if defined(LEGATE_USE_GASNET)

enum CollTag : int {
  BCAST_TAG     = 0,
  GATHER_TAG    = 1,
  ALLTOALL_TAG  = 2,
  ALLTOALLV_TAG = 3,
  MAX_TAG       = 10,
};

#define USE_NEW_COMM

#if defined(USE_NEW_COMM)
static std::vector<MPI_Comm> mpi_comms;
#endif

#else
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
#if defined(LEGATE_USE_GASNET)
  int mpi_rank, mpi_comm_size;
  int *tag_ub, flag, res;
#if defined(USE_NEW_COMM)
  int compare_result;
  MPI_Comm comm = mpi_comms[unique_id];
  res           = MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_result);
  assert(res == MPI_SUCCESS);
  assert(compare_result = MPI_CONGRUENT);
#else
  MPI_Comm comm = MPI_COMM_WORLD;
#endif
  MPI_Comm_get_attr(comm, MPI_TAG_UB, &tag_ub, &flag);
  assert(flag);
  assert(*tag_ub == INT_MAX);
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_comm_size);
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
                         NULL,
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
#if defined(LEGATE_USE_GASNET)
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
                  CollDataType sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  CollDataType recvtype,
                  CollComm global_comm)
{
  log_coll.print("Alltoallv: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAlltoallvMPI(
    sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, global_comm);
#else
  return collAlltoallvLocal(
    sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, global_comm);
#endif
}

int collAlltoall(const void* sendbuf,
                 int sendcount,
                 CollDataType sendtype,
                 void* recvbuf,
                 int recvcount,
                 CollDataType recvtype,
                 CollComm global_comm)
{
  log_coll.print("Alltoall: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAlltoallMPI(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#else
  return collAlltoallLocal(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#endif
}

int collGather(const void* sendbuf,
               int sendcount,
               CollDataType sendtype,
               void* recvbuf,
               int recvcount,
               CollDataType recvtype,
               int root,
               CollComm global_comm)
{
  log_coll.print("Gather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collGatherMPI(
    sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, global_comm);
#else
  printf("Not implemented\n");
  assert(0);
#endif
}

int collAllgather(const void* sendbuf,
                  int sendcount,
                  CollDataType sendtype,
                  void* recvbuf,
                  int recvcount,
                  CollDataType recvtype,
                  CollComm global_comm)
{
  log_coll.print("Allgather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collAllgatherMPI(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#else
  return collAllgatherLocal(
    sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, global_comm);
#endif
}

int collBcast(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  log_coll.print("Bcast: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 global_comm->unique_id,
                 global_comm->global_comm_size);
#if defined(LEGATE_USE_GASNET)
  return collBcast(buf, count, type, root, global_comm);
#else
  printf("Not implemented\n");
  assert(0);
#endif
}

// called from main thread
int collInit(int argc, char* argv[])
{
  current_unique_id = 0;
#if defined(LEGATE_USE_GASNET)
  int provided, res, init_flag = 0;
  MPI_Initialized(&init_flag);
  if (!init_flag) {
    res = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(res == MPI_SUCCESS);
  } else {
    printf(
      "Warning: MPI has been initialized by others, make sure MPI is initialized with "
      "MPI_THREAD_MULTIPLE\n");
  }
#if defined(USE_NEW_COMM)
  mpi_comms.resize(MAX_NB_COMMS, MPI_COMM_NULL);
  for (int i = 0; i < MAX_NB_COMMS; i++) {
    res = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comms[i]);
    assert(res == MPI_SUCCESS);
  }
#endif
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
#if defined(LEGATE_USE_GASNET)
#if defined(USE_NEW_COMM)
  int res;
  for (int i = 0; i < MAX_NB_COMMS; i++) {
    res = MPI_Comm_free(&mpi_comms[i]);
    assert(res == MPI_SUCCESS);
  }
  mpi_comms.clear();
#endif
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
#if defined(USE_NEW_COMM)
  assert(current_unique_id <= MAX_NB_COMMS);
#else
#if defined(LEGATE_USE_GASNET)
  current_unique_id = current_unique_id % MAX_NB_COMMS;
#else
  assert(current_unique_id <= MAX_NB_COMMS);
#endif
#endif
  return CollSuccess;
}

#if defined(LEGATE_USE_GASNET)
MPI_Datatype collDtypeToMPIDtype(CollDataType dtype)
{
  if (dtype == CollDataType::CollInt8) {
    return MPI_INT8_T;
  } else if (dtype == CollDataType::CollChar) {
    return MPI_CHAR;
  } else if (dtype == CollDataType::CollUint8) {
    return MPI_UINT8_T;
  } else if (dtype == CollDataType::CollInt) {
    return MPI_INT;
  } else if (dtype == CollDataType::CollUint32) {
    return MPI_UINT32_T;
  } else if (dtype == CollDataType::CollInt64) {
    return MPI_INT64_T;
  } else if (dtype == CollDataType::CollUint64) {
    return MPI_UINT64_T;
  } else if (dtype == CollDataType::CollFloat) {
    return MPI_FLOAT;
  } else if (dtype == CollDataType::CollDouble) {
    return MPI_DOUBLE;
  } else {
    assert(0);
    return MPI_BYTE;
  }
}

int collGenerateAlltoallTag(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // int send_tag = ((sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALL_TAG) * 10 +
  // global_comm->unique_id; // which dst seg it sends to (in dst rank) int recv_tag = ((global_rank
  // * 10000 + recvfrom_global_rank) * 10 + ALLTOALL_TAG) * 10 + global_comm->unique_id; // idx of
  // current seg we are receving (in src/my rank)
#if defined(USE_NEW_COMM)
  int tag = (rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
#else
#if 1
  int tag = ((rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG) * MAX_NB_COMMS +
            global_comm->unique_id;
#else
  int tag =
    ((rank1 % global_comm->nb_threads * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG) *
      MAX_NB_COMMS +
    global_comm->unique_id;
#endif
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
#if defined(USE_NEW_COMM)
  int tag = (rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
#else
#if 1
  int tag = ((rank1 * 10000 + rank2) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG) * MAX_NB_COMMS +
            global_comm->unique_id;
#else
  int tag = ((rank1 % global_comm->nb_threads * 10000 + rank2) * CollTag::MAX_TAG +
             CollTag::ALLTOALLV_TAG) *
              MAX_NB_COMMS +
            global_comm->unique_id;
#endif
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

int collGenerateBcastTag(int rank, CollComm global_comm)
{
#if defined(USE_NEW_COMM)
  int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
#else
  int tag = (rank * CollTag::MAX_TAG + CollTag::BCAST_TAG) * MAX_NB_COMMS + global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag >= 0);
  return tag;
}

int collGenerateGatherTag(int rank, CollComm global_comm)
{
#if defined(USE_NEW_COMM)
  int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
#else
  int tag = (rank * CollTag::MAX_TAG + CollTag::GATHER_TAG) * MAX_NB_COMMS + global_comm->unique_id;
#endif
  assert(tag < INT_MAX && tag > 0);
  return tag;
}

#else
size_t collGetDtypeSize(CollDataType dtype)
{
  if (dtype == CollDataType::CollInt8 || dtype == CollDataType::CollChar) {
    return sizeof(char);
  } else if (dtype == CollDataType::CollUint8) {
    return sizeof(uint8_t);
  } else if (dtype == CollDataType::CollInt) {
    return sizeof(int);
  } else if (dtype == CollDataType::CollUint32) {
    return sizeof(uint32_t);
  } else if (dtype == CollDataType::CollInt64) {
    return sizeof(int64_t);
  } else if (dtype == CollDataType::CollUint64) {
    return sizeof(uint64_t);
  } else if (dtype == CollDataType::CollFloat) {
    return sizeof(float);
  } else if (dtype == CollDataType::CollDouble) {
    return sizeof(double);
  } else {
    assert(0);
    return -1;
  }
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