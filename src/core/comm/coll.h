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

#ifndef COLL_H
#define COLL_H

//#define DEBUG_PRINT

#include <stddef.h>

#define collSuccess 0
#define collError 1

#if defined (LEGATE_USE_GASNET)
#include <mpi.h>
#define BCAST_TAG     0
#define GATHER_TAG    1
#define ALLTOALL_TAG  2
#define ALLTOALLV_TAG 3

typedef MPI_Datatype collDataType_t;
// TODO: fix it
extern MPI_Datatype collChar;
extern MPI_Datatype collInt8;
extern MPI_Datatype collUint8;
extern MPI_Datatype collInt;
extern MPI_Datatype collUint32;
extern MPI_Datatype collInt64;
extern MPI_Datatype collUint64;
extern MPI_Datatype collFloat;
extern MPI_Datatype collDouble;

typedef struct mapping_table_s {
  int *mpi_rank;
  int *global_rank;
} mapping_table_t;

#else
#include <stdbool.h>

#define MAX_NB_THREADS 128
#define MAX_NB_COMMS 64

typedef struct shared_buffer_s {
  const void* buffers[MAX_NB_THREADS];
  const int* displs[MAX_NB_THREADS];
  bool buffers_ready[MAX_NB_THREADS];
} shared_buffer_t;

typedef struct shared_data_s {
  shared_buffer_t shared_buffer;
  pthread_barrier_t barrier;
  bool ready_flag;
} shared_data_t;

extern shared_data_t shared_data[MAX_NB_COMMS];

typedef enum { 
  collInt8       = 0, collChar       = 0,
  collUint8      = 1,
  collInt        = 2,
  collUint32     = 3,
  collInt64      = 4,
  collUint64     = 5,
  collFloat      = 7,
  collDouble     = 8,
} collDataType_t;
#endif

typedef struct Coll_Comm_s {
#if defined (LEGATE_USE_GASNET)
  MPI_Comm comm;
  mapping_table_t mapping_table;
#else
  volatile shared_buffer_t *shared_buffer;
#endif
  int mpi_rank;
  int mpi_comm_size;
  int global_rank;
  int global_comm_size;
  int unique_id;
  bool status;
} Coll_Comm;

typedef Coll_Comm* collComm_t;

#ifdef __cplusplus
extern "C" {
#endif

int collCommCreate(collComm_t global_comm, int global_comm_size, int global_rank, int unique_id, const int *mapping_table);

int collCommDestroy(collComm_t global_comm);

int collAlltoallv(const void *sendbuf, const int sendcounts[],
                  const int sdispls[], collDataType_t sendtype,
                  void *recvbuf, const int recvcounts[],
                  const int rdispls[], collDataType_t recvtype, 
                  collComm_t global_comm);

int collAlltoall(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                 void *recvbuf, int recvcount, collDataType_t recvtype, 
                 collComm_t global_comm);

int collGather(const void *sendbuf, int sendcount, collDataType_t sendtype, 
               void *recvbuf, int recvcount, collDataType_t recvtype, 
               int root,
               collComm_t global_comm);

int collAllgather(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  collComm_t global_comm);

int collBcast(void *buf, int count, collDataType_t type, 
              int root,
              collComm_t global_comm);

int collInit(int argc, char *argv[]);

int collFinalize(void);

int collGetUniqueId(int* id);

#if defined (LEGATE_USE_GASNET)
int collAlltoallvMPI(const void *sendbuf, const int sendcounts[],
                     const int sdispls[], collDataType_t sendtype,
                     void *recvbuf, const int recvcounts[],
                     const int rdispls[], collDataType_t recvtype, 
                     collComm_t global_comm);

int collAlltoallMPI(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                    void *recvbuf, int recvcount, collDataType_t recvtype, 
                    collComm_t global_comm);

int collGatherMPI(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                  void *recvbuf, int recvcount, collDataType_t recvtype, 
                  int root,
                  collComm_t global_comm);

int collAllgatherMPI(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                     void *recvbuf, int recvcount, collDataType_t recvtype, 
                     collComm_t global_comm);

int collBcastMPI(void *buf, int count, collDataType_t type, 
                 int root,
                 collComm_t global_comm);
#else
size_t get_dtype_size(collDataType_t dtype);

int collAlltoallvLocal(const void *sendbuf, const int sendcounts[],
                       const int sdispls[], collDataType_t sendtype,
                       void *recvbuf, const int recvcounts[],
                       const int rdispls[], collDataType_t recvtype, 
                       collComm_t global_comm);

int collAlltoallLocal(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                      void *recvbuf, int recvcount, collDataType_t recvtype, 
                      collComm_t global_comm);

int collAllgatherLocal(const void *sendbuf, int sendcount, collDataType_t sendtype, 
                       void *recvbuf, int recvcount, collDataType_t recvtype, 
                       collComm_t global_comm);

void collUpdateBuffer(collComm_t global_comm);

void collBarrierLocal(collComm_t global_comm);
#endif

#ifdef __cplusplus
}
#endif

#endif // ifndef COLL_H