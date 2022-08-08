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
#include <unordered_map>

#ifndef LEGATE_USE_GASNET
#include <stdint.h>
#endif

#include "coll.h"
#include "legate.h"
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

static int mpi_tag_ub = 0;

static std::vector<MPI_Comm> mpi_comms;
#else  // undef LEGATE_USE_GASNET
static std::vector<ThreadComm*> thread_comms;
#endif

static int current_unique_id = 0;

static bool coll_inited = false;

// functions start here
#ifdef LEGATE_USE_GASNET
static inline std::pair<int, int> mostFrequent(const int* arr, int n);
static inline int match2ranks(int rank1, int rank2, CollComm global_comm);
#endif

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
  assert(MPI_CONGRUENT == compare_result);

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
  std::pair<int, int> p             = mostFrequent(mapping_table, global_comm_size);
  global_comm->nb_threads           = p.first;
  global_comm->mpi_comm_size_actual = p.second;
#else
  assert(mapping_table == nullptr);
  global_comm->mpi_comm_size        = 1;
  global_comm->mpi_comm_size_actual = 1;
  global_comm->mpi_rank             = 0;
  if (global_comm->global_rank == 0) {
    pthread_barrier_init((pthread_barrier_t*)&(thread_comms[global_comm->unique_id]->barrier),
                         nullptr,
                         global_comm->global_comm_size);
    thread_comms[global_comm->unique_id]->buffers =
      (const void**)malloc(sizeof(void*) * global_comm_size);
    thread_comms[global_comm->unique_id]->displs =
      (const int**)malloc(sizeof(int*) * global_comm_size);
    for (int i = 0; i < global_comm_size; i++) {
      thread_comms[global_comm->unique_id]->buffers[i] = nullptr;
      thread_comms[global_comm->unique_id]->displs[i]  = nullptr;
    }
    __sync_synchronize();
    thread_comms[global_comm->unique_id]->ready_flag = true;
  }
  __sync_synchronize();
  volatile ThreadComm* data = thread_comms[global_comm->unique_id];
  while (data->ready_flag != true) { data = thread_comms[global_comm->unique_id]; }
  global_comm->comm = thread_comms[global_comm->unique_id];
  barrierLocal(global_comm);
  assert(global_comm->comm->ready_flag == true);
  assert(global_comm->comm->buffers != nullptr);
  assert(global_comm->comm->displs != nullptr);
  global_comm->nb_threads = global_comm->global_comm_size;
#endif
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
  barrierLocal(global_comm);
  if (global_comm->global_rank == 0) {
    pthread_barrier_destroy((pthread_barrier_t*)&(thread_comms[global_comm->unique_id]->barrier));
    free(thread_comms[global_comm->unique_id]->buffers);
    thread_comms[global_comm->unique_id]->buffers = nullptr;
    free(thread_comms[global_comm->unique_id]->displs);
    thread_comms[global_comm->unique_id]->displs = nullptr;
    __sync_synchronize();
    thread_comms[global_comm->unique_id]->ready_flag = false;
  }
  __sync_synchronize();
  volatile ThreadComm* data = thread_comms[global_comm->unique_id];
  while (data->ready_flag != false) { data = thread_comms[global_comm->unique_id]; }
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
  // IN_PLACE
  if (sendbuf == recvbuf) {
    log_coll.error("Do not support inplace Alltoallv");
    LEGATE_ABORT;
  }
  log_coll.debug(
    "Alltoallv: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
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
  // IN_PLACE
  if (sendbuf == recvbuf) {
    log_coll.error("Do not support inplace Alltoall");
    LEGATE_ABORT;
  }
  log_coll.debug(
    "Alltoall: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
#ifdef LEGATE_USE_GASNET
  return alltoallMPI(sendbuf, recvbuf, count, type, global_comm);
#else
  return alltoallLocal(sendbuf, recvbuf, count, type, global_comm);
#endif
}

int collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  log_coll.debug(
    "Allgather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
#ifdef LEGATE_USE_GASNET
  return allgatherMPI(sendbuf, recvbuf, count, type, global_comm);
#else
  return allgatherLocal(sendbuf, recvbuf, count, type, global_comm);
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
    log_coll.fatal(
      "MPI has not been initialized, it should be initialized by "
      "GASNet");
    LEGATE_ABORT;
  } else {
    int mpi_thread_model;
    MPI_Query_thread(&mpi_thread_model);
    if (mpi_thread_model != MPI_THREAD_MULTIPLE) {
      log_coll.fatal(
        "MPI has been initialized by others, but is not initialized with "
        "MPI_THREAD_MULTIPLE");
      LEGATE_ABORT;
    }
  }
  // check
  int *tag_ub, flag;
  CHECK_MPI(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag));
  assert(flag);
  mpi_tag_ub = *tag_ub;
  assert(mpi_comms.empty());
#else
  assert(thread_comms.empty());
#endif
  coll_inited = true;
  return CollSuccess;
}

int collFinalize()
{
  assert(coll_inited == true);
  coll_inited = false;
#ifdef LEGATE_USE_GASNET
  for (MPI_Comm& mpi_comm : mpi_comms) { CHECK_MPI(MPI_Comm_free(&mpi_comm)); }
  mpi_comms.clear();
  int fina_flag = 0;
  CHECK_MPI(MPI_Finalized(&fina_flag));
  if (fina_flag == 1) {
    log_coll.fatal("MPI should not have been finalized");
    LEGATE_ABORT;
  }
#else
  for (ThreadComm* thread_comm : thread_comms) {
    assert(!thread_comm->ready_flag);
    free(thread_comm);
  }
  thread_comms.clear();
#endif
  return CollSuccess;
}

int collGetUniqueId(int* id)
{
  *id = current_unique_id;
  current_unique_id++;
  return CollSuccess;
}

int collInitComm()
{
  int id = 0;
  collGetUniqueId(&id);
#ifdef LEGATE_USE_GASNET
#ifdef DEBUG_LEGATE
  int mpi_rank;
  int send_id = id;
  // check if all ranks get the same unique id
  CHECK_MPI(MPI_Bcast(&send_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
  assert(send_id == id);
#endif
  assert(mpi_comms.size() == id);
  // create mpi comm
  MPI_Comm mpi_comm;
  CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
  mpi_comms.push_back(mpi_comm);
#else
  assert(thread_comms.size() == id);
  // create thread comm
  ThreadComm* thread_comm = (ThreadComm*)malloc(sizeof(ThreadComm));
  thread_comm->ready_flag = false;
  thread_comm->buffers    = nullptr;
  thread_comm->displs     = nullptr;
  thread_comms.push_back(thread_comm);
#endif
  log_coll.debug("Init comm id %d", id);
  return id;
}

#ifdef LEGATE_USE_GASNET
static inline std::pair<int, int> mostFrequent(const int* arr, int n)
{
  std::unordered_map<int, int> hash;
  for (int i = 0; i < n; i++) hash[arr[i]]++;

  // find the max frequency
  int max_count = 0;
  std::unordered_map<int, int>::iterator it;
  for (it = hash.begin(); it != hash.end(); it++) {
    if (max_count < it->second) { max_count = it->second; }
  }

  return std::make_pair(max_count, hash.size());
}

static inline int match2ranks(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // send_tag = sendto_global_rank * 10000 + global_rank (concat 2 ranks)
  // which dst seg it sends to (in dst rank)
  // recv_tag = global_rank * 10000 + recvfrom_global_rank (concat 2 ranks)
  // idx of current seg we are receving (in src/my rank)
  // example:
  // 00 | 01 | 02 | 03
  // 10 | 11 | 12 | 13
  // 20 | 21 | 22 | 23
  // 30 | 31 | 32 | 33
  // 01's send_tag = 10, 10's recv_tag = 10, match
  // 12's send_tag = 21, 21's recv_tag = 21, match

  int tag;
  // old tagging system for debug
  // constexpr int const max_ranks = 10000;
  // tag                           = rank1 * max_ranks + rank2;

  // new tagging system, if crash, switch to the old one

  tag = rank1 % global_comm->nb_threads * global_comm->global_comm_size + rank2;

  // Szudzik's Function, two numbers < 32768
  // if (rank1 >= rank2) {
  //   tag = rank1*rank1 + rank1 + rank2;
  // } else {
  //   tag = rank1 + rank2*rank2;
  // }

  // Cantor Pairing Function, two numbers < 32768
  // tag = (rank1 + rank2) * (rank1 + rank2 + 1) / 2 + rank1;

  return tag;
}

MPI_Datatype dtypeToMPIDtype(CollDataType dtype)
{
  switch (dtype) {
    case CollDataType::CollInt8: {
      return MPI_INT8_T;
    }
    case CollDataType::CollChar: {
      return MPI_CHAR;
    }
    case CollDataType::CollUint8: {
      return MPI_UINT8_T;
    }
    case CollDataType::CollInt: {
      return MPI_INT;
    }
    case CollDataType::CollUint32: {
      return MPI_UINT32_T;
    }
    case CollDataType::CollInt64: {
      return MPI_INT64_T;
    }
    case CollDataType::CollUint64: {
      return MPI_UINT64_T;
    }
    case CollDataType::CollFloat: {
      return MPI_FLOAT;
    }
    case CollDataType::CollDouble: {
      return MPI_DOUBLE;
    }
    default: {
      log_coll.fatal("Unknown datatype");
      LEGATE_ABORT;
      return MPI_BYTE;
    }
  }
}

int generateAlltoallTag(int rank1, int rank2, CollComm global_comm)
{
  int tag = match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int generateAlltoallvTag(int rank1, int rank2, CollComm global_comm)
{
  int tag = match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int generateBcastTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
  assert(tag <= mpi_tag_ub && tag >= 0);
  return tag;
}

int generateGatherTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

#else  // undef LEGATE_USE_GASNET
size_t getDtypeSize(CollDataType dtype)
{
  switch (dtype) {
    case CollDataType::CollInt8:
    case CollDataType::CollChar: {
      return sizeof(char);
    }
    case CollDataType::CollUint8: {
      return sizeof(uint8_t);
    }
    case CollDataType::CollInt: {
      return sizeof(int);
    }
    case CollDataType::CollUint32: {
      return sizeof(uint32_t);
    }
    case CollDataType::CollInt64: {
      return sizeof(int64_t);
    }
    case CollDataType::CollUint64: {
      return sizeof(uint64_t);
    }
    case CollDataType::CollFloat: {
      return sizeof(float);
    }
    case CollDataType::CollDouble: {
      return sizeof(double);
    }
    default: {
      log_coll.fatal("Unknown datatype");
      LEGATE_ABORT;
      return 0;
    }
  }
}

void resetLocalBuffer(CollComm global_comm)
{
  int global_rank                         = global_comm->global_rank;
  global_comm->comm->buffers[global_rank] = nullptr;
  global_comm->comm->displs[global_rank]  = nullptr;
}

void barrierLocal(CollComm global_comm)
{
  assert(coll_inited == true);
  pthread_barrier_wait(const_cast<pthread_barrier_t*>(&(global_comm->comm->barrier)));
}
#endif

void* allocateInplaceBuffer(const void* recvbuf, size_t size)
{
  void* sendbuf_tmp = malloc(size);
  assert(sendbuf_tmp != nullptr);
  memcpy(sendbuf_tmp, recvbuf, size);
  return sendbuf_tmp;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate

extern "C" {

void legate_cpucoll_finalize(void) { legate::comm::coll::collFinalize(); }

int legate_cpucoll_initcomm(void) { return legate::comm::coll::collInitComm(); }
}