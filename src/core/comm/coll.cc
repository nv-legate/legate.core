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

BackendNetwork* backend_network = nullptr;

// functions start here
int collCommCreate(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table)
{
  return backend_network->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

int collCommDestroy(CollComm global_comm) { return backend_network->comm_destroy(global_comm); }

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
  return backend_network->alltoallv(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
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
  return backend_network->alltoall(sendbuf, recvbuf, count, type, global_comm);
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
  return backend_network->allgather(sendbuf, recvbuf, count, type, global_comm);
}

// called from main thread
int collInit(int argc, char* argv[])
{
#ifdef LEGATE_USE_NETWORK
  char* network    = getenv("LEGATE_NEED_NETWORK");
  int need_network = 0;
  if (network != nullptr) { need_network = atoi(network); }
  if (need_network) {
    backend_network = new MPINetwork(argc, argv);
  } else {
    backend_network = new LocalNetwork(argc, argv);
  }
#else
  backend_network = new LocalNetwork(argc, argv);
#endif
  return CollSuccess;
}

int collFinalize()
{
  delete backend_network;
  return CollSuccess;
}

int collInitComm() { return backend_network->init_comm(); }

BackendNetwork::BackendNetwork() : coll_inited(false), current_unique_id(0) {}

BackendNetwork::~BackendNetwork() {}

int BackendNetwork::collGetUniqueId(int* id)
{
  *id = current_unique_id;
  current_unique_id++;
  return CollSuccess;
}

void* BackendNetwork::allocateInplaceBuffer(const void* recvbuf, size_t size)
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
