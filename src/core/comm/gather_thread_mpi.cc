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
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

using namespace Legion;
extern Logger log_coll;

int gatherMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // Should not see inplace here
  if (sendbuf == recvbuf) { assert(0); }

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  // non-root
  if (global_rank != root) {
    tag = generateGatherTag(global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("GatherMPI: non-root send global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   global_rank,
                   global_comm->mpi_rank,
                   root,
                   root_mpi_rank,
                   tag);
#endif
    CHECK_MPI(MPI_Send(sendbuf, count, mpi_type, root_mpi_rank, tag, global_comm->comm));
    return CollSuccess;
  }

  // root
  MPI_Aint incr, lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);
  incr      = type_extent * static_cast<ptrdiff_t>(count);
  char* dst = static_cast<char*>(recvbuf);
  int recvfrom_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = generateGatherTag(i, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "GatherMPI: root i %d === global_rank %d, mpi rank %d, recv %p, from %d (%d), tag %d",
      i,
      global_rank,
      global_comm->mpi_rank,
      dst,
      i,
      recvfrom_mpi_rank,
      tag);
#endif
    assert(dst != nullptr);
    if (global_rank == i) {
      memcpy(dst, sendbuf, incr);
    } else {
      CHECK_MPI(MPI_Recv(dst, count, mpi_type, recvfrom_mpi_rank, tag, global_comm->comm, &status));
    }
    dst += incr;
  }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate