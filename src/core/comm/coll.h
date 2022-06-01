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

#if defined(LEGATE_USE_GASNET)
#include <mpi.h>
#endif

namespace legate {
namespace comm {
namespace coll {

#if defined(LEGATE_USE_GASNET)

struct RankMappingTable {
  int* mpi_rank;
  int* global_rank;
};

#else

struct ThreadComm {
  pthread_barrier_t barrier;
  bool ready_flag;
  const void** buffers;
  const int** displs;
};
#endif

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

struct Coll_Comm {
#if defined(LEGATE_USE_GASNET)
  MPI_Comm comm;
  RankMappingTable mapping_table;
#else
  volatile ThreadComm* comm;
#endif
  int mpi_rank;
  int mpi_comm_size;
  int global_rank;
  int global_comm_size;
  int nb_threads;
  int unique_id;
  bool status;
};

typedef Coll_Comm* CollComm;

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

int collGather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm);

int collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int collBcast(void* buf, int count, CollDataType type, int root, CollComm global_comm);

int collInit(int argc, char* argv[]);

int collFinalize(void);

int collGetUniqueId(int* id);

// The following functions should not be called by users
#if defined(LEGATE_USE_GASNET)
int alltoallvMPI(const void* sendbuf,
                 const int sendcounts[],
                 const int sdispls[],
                 void* recvbuf,
                 const int recvcounts[],
                 const int rdispls[],
                 CollDataType type,
                 CollComm global_comm);

int alltoallMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int gatherMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm);

int allgatherMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int bcastMPI(void* buf, int count, CollDataType type, int root, CollComm global_comm);

MPI_Datatype collDtypeToMPIDtype(CollDataType dtype);

int collGenerateAlltoallTag(int rank1, int rank2, CollComm global_comm);

int collGenerateAlltoallvTag(int rank1, int rank2, CollComm global_comm);

int collGenerateBcastTag(int rank, CollComm global_comm);

int collGenerateGatherTag(int rank, CollComm global_comm);
#else
size_t collGetDtypeSize(CollDataType dtype);

int alltoallvLocal(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   CollDataType type,
                   CollComm global_comm);

int alltoallLocal(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

int allgatherLocal(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

void collUpdateBuffer(CollComm global_comm);

void collBarrierLocal(CollComm global_comm);
#endif

void* collAllocateInplaceBuffer(const void* recvbuf, size_t size);

}  // namespace coll
}  // namespace comm
}  // namespace legate