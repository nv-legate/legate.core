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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coll.h"
#include "legate.h"
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

using namespace Legion;
extern Logger log_coll;

// public functions start from here

LocalNetwork::LocalNetwork(int argc, char* argv[]) : BackendNetwork()
{
  log_coll.debug("Enable LocalNetwork");
  assert(current_unique_id == 0);
  assert(thread_comms.empty());
  BackendNetwork::coll_inited = true;
  BackendNetwork::comm_type   = CollCommType::CollLocal;
}

LocalNetwork::~LocalNetwork()
{
  log_coll.debug("Finalize LocalNetwork");
  assert(BackendNetwork::coll_inited == true);
  for (ThreadComm* thread_comm : thread_comms) {
    assert(!thread_comm->ready_flag);
    free(thread_comm);
  }
  thread_comms.clear();
  BackendNetwork::coll_inited = false;
}

int LocalNetwork::comm_create(CollComm global_comm,
                              int global_comm_size,
                              int global_rank,
                              int unique_id,
                              const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
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
  global_comm->local_comm = thread_comms[global_comm->unique_id];
  barrierLocal(global_comm);
  assert(global_comm->local_comm->ready_flag == true);
  assert(global_comm->local_comm->buffers != nullptr);
  assert(global_comm->local_comm->displs != nullptr);
  global_comm->nb_threads = global_comm->global_comm_size;
  return CollSuccess;
}

int LocalNetwork::comm_destroy(CollComm global_comm)
{
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
  global_comm->status = false;
  return CollSuccess;
}

int LocalNetwork::init_comm()
{
  int id = 0;
  collGetUniqueId(&id);
  assert(thread_comms.size() == id);
  // create thread comm
  ThreadComm* thread_comm = (ThreadComm*)malloc(sizeof(ThreadComm));
  thread_comm->ready_flag = false;
  thread_comm->buffers    = nullptr;
  thread_comm->displs     = nullptr;
  thread_comms.push_back(thread_comm);
  log_coll.debug("Init comm id %d", id);
  return id;
}

int LocalNetwork::alltoallv(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            CollDataType type,
                            CollComm global_comm)
{
  int res;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  global_comm->local_comm->displs[global_rank]  = sdispls;
  global_comm->local_comm->buffers[global_rank] = sendbuf;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id  = global_rank;
  const void* src_base = nullptr;
  const int* displs    = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr ||
           global_comm->local_comm->displs[recvfrom_global_rank] == nullptr)
      ;
    src_base  = global_comm->local_comm->buffers[recvfrom_global_rank];
    displs    = global_comm->local_comm->displs[recvfrom_global_rank];
    char* src = static_cast<char*>(const_cast<void*>(src_base)) +
                static_cast<ptrdiff_t>(displs[recvfrom_seg_id]) * type_extent;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallvLocal i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, sdispls %d, %p) to "
      "rank %d (seg "
      "%d, rdispls %d, %p)",
      i,
      global_rank,
      type_extent,
      recvfrom_global_rank,
      recvfrom_seg_id,
      sdispls[recvfrom_seg_id],
      src,
      global_rank,
      recvfrom_global_rank,
      rdispls[recvfrom_global_rank],
      dst);
#endif
    memcpy(dst, src, recvcounts[recvfrom_global_rank] * type_extent);
  }

  barrierLocal(global_comm);

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

int LocalNetwork::alltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  int res;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  global_comm->local_comm->buffers[global_rank] = sendbuf;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id  = global_rank;
  const void* src_base = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr)
      ;
    src_base  = global_comm->local_comm->buffers[recvfrom_global_rank];
    char* src = static_cast<char*>(const_cast<void*>(src_base)) +
                static_cast<ptrdiff_t>(recvfrom_seg_id) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallLocal i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, %p) to rank %d (seg "
      "%d, %p)",
      i,
      global_rank,
      type_extent,
      recvfrom_global_rank,
      recvfrom_seg_id,
      src,
      global_rank,
      recvfrom_global_rank,
      dst);
#endif
    memcpy(dst, src, count * type_extent);
  }

  barrierLocal(global_comm);

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

int LocalNetwork::allgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  const void* sendbuf_tmp = sendbuf;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) { sendbuf_tmp = allocateInplaceBuffer(recvbuf, type_extent * count); }

  global_comm->local_comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  for (int recvfrom_global_rank = 0; recvfrom_global_rank < total_size; recvfrom_global_rank++) {
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr)
      ;
    const void* src = global_comm->local_comm->buffers[recvfrom_global_rank];
    char* dst       = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AllgatherLocal i: %d === global_rank %d, dtype %d, copy rank %d (%p) to rank %d (%p)",
      recvfrom_global_rank,
      global_rank,
      type_extent,
      recvfrom_global_rank,
      src,
      global_rank,
      dst);
#endif
    memcpy(dst, src, count * type_extent);
  }

  barrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(const_cast<void*>(sendbuf_tmp)); }

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

// protected functions start from here

size_t LocalNetwork::getDtypeSize(CollDataType dtype)
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

void LocalNetwork::resetLocalBuffer(CollComm global_comm)
{
  int global_rank                               = global_comm->global_rank;
  global_comm->local_comm->buffers[global_rank] = nullptr;
  global_comm->local_comm->displs[global_rank]  = nullptr;
}

void LocalNetwork::barrierLocal(CollComm global_comm)
{
  assert(BackendNetwork::coll_inited == true);
  pthread_barrier_wait(const_cast<pthread_barrier_t*>(&(global_comm->local_comm->barrier)));
}

}  // namespace coll
}  // namespace comm
}  // namespace legate