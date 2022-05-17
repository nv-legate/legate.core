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

namespace legate {
namespace comm {
namespace coll {

int collGatherMPI(const void* sendbuf,
                  int sendcount,
                  CollDataType sendtype,
                  void* recvbuf,
                  int recvcount,
                  CollDataType recvtype,
                  int root,
                  CollComm global_comm)
{
  int res;

  // int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  int total_size = global_comm->global_comm_size;
  MPI_Status status;

  int global_rank = global_comm->global_rank;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) { assert(0); }

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  // non-root
  if (global_rank != root) {
    tag = collGenerateGatherTag(global_rank, global_comm);
#ifdef DEBUG_PRINT
    printf("Gather Send global_rank %d, mpi rank %d, send to %d (%d), tag %d\n",
           global_rank,
           global_comm->mpi_rank,
           root,
           root_mpi_rank,
           tag);
#endif
    return MPI_Send(sendbuf, sendcount, sendtype, root_mpi_rank, tag, global_comm->comm);
  }

  // root
  MPI_Aint incr, lb, recvtype_extent;
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
  incr      = recvtype_extent * (ptrdiff_t)recvcount;
  char* dst = (char*)recvbuf;
  int recvfrom_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = collGenerateGatherTag(i, global_comm);
#ifdef DEBUG_PRINT
    printf("Gather i: %d === global_rank %d, mpi rank %d, recv %p, from %d (%d), tag %d\n",
           i,
           global_rank,
           global_comm->mpi_rank,
           dst,
           i,
           recvfrom_mpi_rank,
           tag);
#endif
    assert(dst != NULL);
    if (global_rank == i) {
      memcpy(dst, sendbuf, incr);
    } else {
      res = MPI_Recv(dst, recvcount, recvtype, recvfrom_mpi_rank, tag, global_comm->comm, &status);
      assert(res == MPI_SUCCESS);
    }
    dst += incr;
  }

  return collSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate