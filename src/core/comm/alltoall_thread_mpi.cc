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

static int alltoallMPIInplace(void* recvbuf,
                              int recvcount,
                              MPI_Datatype recvtype,
                              CollComm global_comm)
{
  int total_size = global_comm->global_comm_size;
  MPI_Status status;
  MPI_Request request;

  MPI_Aint lb, recvtype_extent;
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);
  size_t max_size    = recvtype_extent * recvcount;
  size_t packed_size = 0;

  char* tmp_buffer = (char*)malloc(sizeof(char) * max_size);
  assert(tmp_buffer != nullptr);

  int global_rank = global_comm->global_rank;

  int right, left, right_mpi_rank, left_mpi_rank, send_tag, recv_tag;
  for (int i = 1; i <= (total_size >> 1); ++i) {
    right          = (global_rank + i) % total_size;
    left           = (global_rank + total_size - i) % total_size;
    right_mpi_rank = global_comm->mapping_table.mpi_rank[right];
    left_mpi_rank  = global_comm->mapping_table.mpi_rank[left];
    assert(right == global_comm->mapping_table.global_rank[right]);
    assert(left == global_comm->mapping_table.global_rank[left]);

    char* send_tmp_buffer = (char*)recvbuf + right * recvcount * recvtype_extent;
    memcpy(tmp_buffer, send_tmp_buffer, recvcount * recvtype_extent);
    packed_size = max_size;

    // receive data from the right
    recv_tag = collGenerateAlltoallTag(global_rank, right, global_comm);
    CHECK_MPI(MPI_Irecv((char*)recvbuf + right * recvcount * recvtype_extent,
                        recvcount,
                        recvtype,
                        right_mpi_rank,
                        recv_tag,
                        global_comm->comm,
                        &request));

    if (left != right) {
      // send data to the left
      send_tag = collGenerateAlltoallTag(left, global_rank, global_comm);
      CHECK_MPI(MPI_Send((char*)recvbuf + left * recvcount * recvtype_extent,
                         recvcount,
                         recvtype,
                         left_mpi_rank,
                         send_tag,
                         global_comm->comm));

      CHECK_MPI(MPI_Wait(&request, MPI_STATUSES_IGNORE));

      // receive data from the left
      recv_tag = collGenerateAlltoallTag(global_rank, left, global_comm);
      CHECK_MPI(MPI_Irecv((char*)recvbuf + left * recvcount * recvtype_extent,
                          recvcount,
                          recvtype,
                          left_mpi_rank,
                          recv_tag,
                          global_comm->comm,
                          &request));
    }

    // send data to the right
    assert(packed_size == recvtype_extent * recvcount);
    send_tag = collGenerateAlltoallTag(right, global_rank, global_comm);
    CHECK_MPI(
      MPI_Send(tmp_buffer, packed_size, MPI_PACKED, right_mpi_rank, send_tag, global_comm->comm));

    CHECK_MPI(MPI_Wait(&request, MPI_STATUSES_IGNORE));
  }

  free(tmp_buffer);

  return CollSuccess;
}

int alltoallMPI(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = collDtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);

  void* sendbuf_tmp = const_cast<void*>(sendbuf);

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    // Not sure which way is better
    // return alltoallMPIInplace(recvbuf, recvcount, mpi_type, global_comm);
    sendbuf_tmp = collAllocateInplaceBuffer(recvbuf, total_size * type_extent * count);
  }

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(sendbuf_tmp) +
                static_cast<ptrdiff_t>(sendto_global_rank) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = collGenerateAlltoallTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = collGenerateAlltoallTag(global_rank, recvfrom_global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallMPI i: %d === global_rank %d, mpi rank %d, send %d to %d, send_tag %d, recv %d from "
      "%d, "
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

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate