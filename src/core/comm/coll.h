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

//#define DEBUG_PRINT

#include <stdbool.h>
#include <stddef.h>

#if defined(LEGATE_USE_GASNET)
#include <mpi.h>
#endif

#define collSuccess 0
#define collError 1

#define MAX_NB_COMMS 100

namespace legate {
namespace comm {
namespace coll {

#if defined(LEGATE_USE_GASNET)

#define BCAST_TAG 0
#define GATHER_TAG 1
#define ALLTOALL_TAG 2
#define ALLTOALLV_TAG 3

#define MAX_COLL_TYPES 10

typedef MPI_Datatype CollDataType;
// TODO: fix it
extern MPI_Datatype CollChar;
extern MPI_Datatype CollInt8;
extern MPI_Datatype CollUint8;
extern MPI_Datatype CollInt;
extern MPI_Datatype CollUint32;
extern MPI_Datatype CollInt64;
extern MPI_Datatype CollUint64;
extern MPI_Datatype CollFloat;
extern MPI_Datatype CollDouble;

struct RankMappingTable {
  int* mpi_rank;
  int* global_rank;
};

#else

#define MAX_NB_THREADS 128

struct ThreadSharedData {
  void* buffers[MAX_NB_THREADS];
  int* displs[MAX_NB_THREADS];
  pthread_barrier_t barrier;
  bool ready_flag;
};

extern volatile ThreadSharedData shared_data[MAX_NB_COMMS];

typedef enum {
  CollInt8   = 0,
  CollChar   = 0,
  CollUint8  = 1,
  CollInt    = 2,
  CollUint32 = 3,
  CollInt64  = 4,
  CollUint64 = 5,
  CollFloat  = 7,
  CollDouble = 8,
} CollDataType;
#endif

typedef struct Coll_Comm_s {
#if defined(LEGATE_USE_GASNET)
  MPI_Comm comm;
  RankMappingTable mapping_table;
#else
  volatile ThreadSharedData* shared_data;
#endif
  int mpi_rank;
  int mpi_comm_size;
  int global_rank;
  int global_comm_size;
  int nb_threads;
  int unique_id;
  bool status;
} Coll_Comm;

typedef Coll_Comm* collComm_t;

int collCommCreate(collComm_t global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table);

int collCommDestroy(collComm_t global_comm);

int collAlltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  CollDataType sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  CollDataType recvtype,
                  collComm_t global_comm);

int collAlltoall(const void* sendbuf,
                 int sendcount,
                 CollDataType sendtype,
                 void* recvbuf,
                 int recvcount,
                 CollDataType recvtype,
                 collComm_t global_comm);

int collGather(const void* sendbuf,
               int sendcount,
               CollDataType sendtype,
               void* recvbuf,
               int recvcount,
               CollDataType recvtype,
               int root,
               collComm_t global_comm);

int collAllgather(const void* sendbuf,
                  int sendcount,
                  CollDataType sendtype,
                  void* recvbuf,
                  int recvcount,
                  CollDataType recvtype,
                  collComm_t global_comm);

int collBcast(void* buf, int count, CollDataType type, int root, collComm_t global_comm);

int collInit(int argc, char* argv[]);

int collFinalize(void);

int collGetUniqueId(int* id);

// The following functions should not be called by users
#if defined(LEGATE_USE_GASNET)
int collAlltoallvMPI(const void* sendbuf,
                     const int sendcounts[],
                     const int sdispls[],
                     CollDataType sendtype,
                     void* recvbuf,
                     const int recvcounts[],
                     const int rdispls[],
                     CollDataType recvtype,
                     collComm_t global_comm);

int collAlltoallMPI(const void* sendbuf,
                    int sendcount,
                    CollDataType sendtype,
                    void* recvbuf,
                    int recvcount,
                    CollDataType recvtype,
                    collComm_t global_comm);

int collGatherMPI(const void* sendbuf,
                  int sendcount,
                  CollDataType sendtype,
                  void* recvbuf,
                  int recvcount,
                  CollDataType recvtype,
                  int root,
                  collComm_t global_comm);

int collAllgatherMPI(const void* sendbuf,
                     int sendcount,
                     CollDataType sendtype,
                     void* recvbuf,
                     int recvcount,
                     CollDataType recvtype,
                     collComm_t global_comm);

int collBcastMPI(void* buf, int count, CollDataType type, int root, collComm_t global_comm);

int collGenerateAlltoallTag(int rank1, int rank2, collComm_t global_comm);

int collGenerateAlltoallvTag(int rank1, int rank2, collComm_t global_comm);

int collGenerateBcastTag(int rank, collComm_t global_comm);

int collGenerateGatherTag(int rank, collComm_t global_comm);
#else
size_t get_dtype_size(CollDataType dtype);

int collAlltoallvLocal(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       CollDataType sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       CollDataType recvtype,
                       collComm_t global_comm);

int collAlltoallLocal(const void* sendbuf,
                      int sendcount,
                      CollDataType sendtype,
                      void* recvbuf,
                      int recvcount,
                      CollDataType recvtype,
                      collComm_t global_comm);

int collAllgatherLocal(const void* sendbuf,
                       int sendcount,
                       CollDataType sendtype,
                       void* recvbuf,
                       int recvcount,
                       CollDataType recvtype,
                       collComm_t global_comm);

void collUpdateBuffer(collComm_t global_comm);

void collBarrierLocal(collComm_t global_comm);
#endif

}  // namespace coll
}  // namespace comm
}  // namespace legate