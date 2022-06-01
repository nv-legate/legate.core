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

int allgatherMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);

  void* sendbuf_tmp = const_cast<void*>(sendbuf);

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) { sendbuf_tmp = allocateInplaceBuffer(recvbuf, type_extent * count); }

  gatherMPI(sendbuf_tmp, recvbuf, count, type, 0, global_comm);

  bcastMPI(recvbuf, count * total_size, type, 0, global_comm);

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate