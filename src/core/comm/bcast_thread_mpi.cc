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

int bcastMPI(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  int tag;
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // non-root
  if (global_rank != root) {
    tag = generateBcastTag(global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("BcastMPI: non-root recv global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   global_rank,
                   global_comm->mpi_rank,
                   root,
                   root_mpi_rank,
                   tag);
#endif
    CHECK_MPI(MPI_Recv(buf, count, mpi_type, root_mpi_rank, tag, global_comm->comm, &status));
    return CollSuccess;
  }

  // root
  int sendto_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = generateBcastTag(i, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("BcastMPI: root i %d === global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   i,
                   global_rank,
                   global_comm->mpi_rank,
                   i,
                   sendto_mpi_rank,
                   tag);
#endif
    if (global_rank != i) {
      CHECK_MPI(MPI_Send(buf, count, mpi_type, sendto_mpi_rank, tag, global_comm->comm));
    }
  }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate