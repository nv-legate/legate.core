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
#include <algorithm>

#include "coll.h"
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

using namespace Legion;
extern Logger log_coll;

int sendMPI(
  const void* sendbuf, int count, CollDataType type, int dest, int tag, CollComm global_comm)
{
  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  int dest_mpi_rank = global_comm->mapping_table.mpi_rank[dest];
  int send_tag      = generateP2PTag(tag);
#ifdef DEBUG_LEGATE
  log_coll.debug("sendMPI global_rank %d, mpi rank %d, send to %d (%d), send_tag %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 dest,
                 dest_mpi_rank,
                 send_tag);
#endif
  CHECK_MPI(MPI_Send(sendbuf, count, mpi_type, dest_mpi_rank, send_tag, global_comm->comm));

  return CollSuccess;
}

int recvMPI(void* recvbuf, int count, CollDataType type, int source, int tag, CollComm global_comm)
{
  MPI_Status status;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  int source_mpi_rank = global_comm->mapping_table.mpi_rank[source];
  int recv_tag        = generateP2PTag(tag);
#ifdef DEBUG_LEGATE
  log_coll.debug("recvMPI global_rank %d, mpi rank %d, recv from %d (%d), recv_tag %d",
                 global_comm->global_rank,
                 global_comm->mpi_rank,
                 source,
                 source_mpi_rank,
                 recv_tag);
#endif
  CHECK_MPI(
    MPI_Recv(recvbuf, count, mpi_type, source_mpi_rank, recv_tag, global_comm->comm, &status));

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate