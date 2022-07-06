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

int alltoallMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(const_cast<void*>(sendbuf)) +
                static_cast<ptrdiff_t>(sendto_global_rank) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = generateAlltoallTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = generateAlltoallTag(global_rank, recvfrom_global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallMPI i: %d === global_rank %d, mpi rank %d, send to %d (%d), send_tag %d, "
      "recv from %d (%d), "
      "recv_tag %d",
      i,
      global_rank,
      global_comm->mpi_rank,
      sendto_global_rank,
      sendto_mpi_rank,
      send_tag,
      recvfrom_global_rank,
      recvfrom_mpi_rank,
      recv_tag);
#endif
    CHECK_MPI(MPI_Sendrecv(src,
                           count,
                           mpi_type,
                           sendto_mpi_rank,
                           send_tag,
                           dst,
                           count,
                           mpi_type,
                           recvfrom_mpi_rank,
                           recv_tag,
                           global_comm->comm,
                           &status));
  }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate