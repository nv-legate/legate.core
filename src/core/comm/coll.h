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

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <vector>

#ifdef LEGATE_USE_NETWORK
#include <mpi.h>
#endif

// If we aren't building with networking, we'll use pthread_barrier to
// construct a communicator for thread-local communication. Mac OS
// does not implement pthread barriers, so we need to include an
// implementation in case they are not defined. We also need to
// include unistd.h since that defines _POSIX_BARRIERS.
#include <unistd.h>
#if !defined(_POSIX_BARRIERS) || (_POSIX_BARRIERS < 0)
#include "core/comm/pthread_barrier.h"
#endif

namespace legate {
namespace comm {
namespace coll {

#ifdef LEGATE_USE_NETWORK
struct RankMappingTable {
  int* mpi_rank;
  int* global_rank;
};
#endif

struct ThreadComm {
  pthread_barrier_t barrier;
  bool ready_flag;
  const void** buffers;
  const int** displs;
};

enum class CollDataType : int {
  CollInt8   = 0,
  CollChar   = 1,
  CollUint8  = 2,
  CollInt    = 3,
  CollUint32 = 4,
  CollInt64  = 5,
  CollUint64 = 6,
  CollFloat  = 7,
  CollDouble = 8,
};

enum CollStatus : int {
  CollSuccess = 0,
  CollError   = 1,
};

enum CollCommType : int {
  CollMPI   = 0,
  CollLocal = 1,
};

struct Coll_Comm {
#ifdef LEGATE_USE_NETWORK
  MPI_Comm mpi_comm;
  RankMappingTable mapping_table;
#endif
  volatile ThreadComm* local_comm;
  int mpi_rank;
  int mpi_comm_size;
  int mpi_comm_size_actual;
  int global_rank;
  int global_comm_size;
  int nb_threads;
  int unique_id;
  bool status;
};

typedef Coll_Comm* CollComm;

class BackendNetwork {
 public:
  BackendNetwork();
  virtual ~BackendNetwork();
  virtual int init_comm() = 0;

  virtual int comm_create(CollComm global_comm,
                          int global_comm_size,
                          int global_rank,
                          int unique_id,
                          const int* mapping_table) = 0;

  virtual int comm_destroy(CollComm global_comm) = 0;

  virtual int alltoallv(const void* sendbuf,
                        const int sendcounts[],
                        const int sdispls[],
                        void* recvbuf,
                        const int recvcounts[],
                        const int rdispls[],
                        CollDataType type,
                        CollComm global_comm) = 0;

  virtual int alltoall(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm) = 0;

  virtual int allgather(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm) = 0;

 protected:
  int collGetUniqueId(int* id);

  void* allocateInplaceBuffer(const void* recvbuf, size_t size);

 public:
  CollCommType comm_type;

 protected:
  bool coll_inited;
  int current_unique_id;
};

#ifdef LEGATE_USE_NETWORK
class MPINetwork : public BackendNetwork {
 public:
  MPINetwork(int argc, char* argv[]);

  ~MPINetwork();

  int init_comm();

  int comm_create(CollComm global_comm,
                  int global_comm_size,
                  int global_rank,
                  int unique_id,
                  const int* mapping_table);

  int comm_destroy(CollComm global_comm);

  int alltoallv(const void* sendbuf,
                const int sendcounts[],
                const int sdispls[],
                void* recvbuf,
                const int recvcounts[],
                const int rdispls[],
                CollDataType type,
                CollComm global_comm);

  int alltoall(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

  int allgather(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

 protected:
  int gather(const void* sendbuf,
             void* recvbuf,
             int count,
             CollDataType type,
             int root,
             CollComm global_comm);

  int bcast(void* buf, int count, CollDataType type, int root, CollComm global_comm);

  MPI_Datatype dtypeToMPIDtype(CollDataType dtype);

  int generateAlltoallTag(int rank1, int rank2, CollComm global_comm);

  int generateAlltoallvTag(int rank1, int rank2, CollComm global_comm);

  int generateBcastTag(int rank, CollComm global_comm);

  int generateGatherTag(int rank, CollComm global_comm);

 private:
  int mpi_tag_ub;
  bool self_init_mpi;
  std::vector<MPI_Comm> mpi_comms;
};
#endif

class LocalNetwork : public BackendNetwork {
 public:
  LocalNetwork(int argc, char* argv[]);

  ~LocalNetwork();

  int init_comm();

  int comm_create(CollComm global_comm,
                  int global_comm_size,
                  int global_rank,
                  int unique_id,
                  const int* mapping_table);

  int comm_destroy(CollComm global_comm);

  int alltoallv(const void* sendbuf,
                const int sendcounts[],
                const int sdispls[],
                void* recvbuf,
                const int recvcounts[],
                const int rdispls[],
                CollDataType type,
                CollComm global_comm);

  int alltoall(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

  int allgather(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

 protected:
  size_t getDtypeSize(CollDataType dtype);

  void resetLocalBuffer(CollComm global_comm);

  void barrierLocal(CollComm global_comm);

 private:
  std::vector<ThreadComm*> thread_comms;
};

extern BackendNetwork* backend_network;

int collCommCreate(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table);

int collCommDestroy(CollComm global_comm);

int collAlltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  CollDataType type,
                  CollComm global_comm);

int collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int collInit(int argc, char* argv[]);

int collFinalize();

int collInitComm();

}  // namespace coll
}  // namespace comm
}  // namespace legate
