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

#ifndef LEGATE_USE_NETWORK
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

#ifdef LEGATE_USE_NETWORK

#else
static std::vector<ThreadComm*> thread_comms;
#endif
static int current_unique_id = 0;

static bool coll_inited = false;

BackendNetwork* backend_network = nullptr;

// functions start here

int collCommCreate(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table)
{
#ifdef LEGATE_USE_NETWORK
  return backend_network->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
#else
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
  global_comm->comm = thread_comms[global_comm->unique_id];
  barrierLocal(global_comm);
  assert(global_comm->comm->ready_flag == true);
  assert(global_comm->comm->buffers != nullptr);
  assert(global_comm->comm->displs != nullptr);
  global_comm->nb_threads = global_comm->global_comm_size;
  return CollSuccess;
#endif
}

int collCommDestroy(CollComm global_comm)
{
#ifdef LEGATE_USE_NETWORK
  return backend_network->comm_destroy(global_comm);
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
  global_comm->status = false;
  return CollSuccess;
#endif
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
#ifdef LEGATE_USE_NETWORK
  return backend_network->alltoallv(
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
#ifdef LEGATE_USE_NETWORK
  return backend_network->alltoall(sendbuf, recvbuf, count, type, global_comm);
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
#ifdef LEGATE_USE_NETWORK
  return backend_network->allgather(sendbuf, recvbuf, count, type, global_comm);
#else
  return allgatherLocal(sendbuf, recvbuf, count, type, global_comm);
#endif
}

// called from main thread
int collInit(int argc, char* argv[])
{
  current_unique_id = 0;
#ifdef LEGATE_USE_NETWORK
  backend_network = new MPINetwork(argc, argv);
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
#ifdef LEGATE_USE_NETWORK
  delete backend_network;
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
#ifdef LEGATE_USE_NETWORK
  return backend_network->init_comm();
#else
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
#endif
}

#ifdef LEGATE_USE_NETWORK

#else  // undef LEGATE_USE_NETWORK
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
