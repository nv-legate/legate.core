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

static int collAlltoallvMPIInplace(void* recvbuf,
                                   const int recvcounts[],
                                   const int rdispls[],
                                   MPI_Datatype recvtype,
                                   CollComm global_comm)
{
  int res;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;
  MPI_Status status;
  MPI_Request request;

  MPI_Aint lb, recvtype_extent;
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);

  size_t max_size    = 0;
  size_t packed_size = 0;
  for (int i = 0; i < total_size; ++i) {
    if (i == global_rank) { continue; }
    packed_size = recvcounts[i] * recvtype_extent;
    max_size    = std::max(packed_size, max_size);
  }

  // Easy way out
  if ((1 == total_size) || (0 == max_size)) { return CollSuccess; }

  char* tmp_buffer = (char*)malloc(sizeof(char) * max_size);
  assert(tmp_buffer != nullptr);

  int right, left, right_mpi_rank, left_mpi_rank, send_tag, recv_tag;
  for (int i = 1; i <= (total_size >> 1); ++i) {
    right          = (global_rank + i) % total_size;
    left           = (global_rank + total_size - i) % total_size;
    right_mpi_rank = global_comm->mapping_table.mpi_rank[right];
    left_mpi_rank  = global_comm->mapping_table.mpi_rank[left];
    assert(right == global_comm->mapping_table.global_rank[right]);
    assert(left == global_comm->mapping_table.global_rank[left]);

    if (0 != recvcounts[right]) { /* nothing to exchange with the peer on the right */

      char* send_tmp_buffer = (char*)recvbuf + rdispls[right] * recvtype_extent;
      memcpy(tmp_buffer, send_tmp_buffer, recvcounts[right] * recvtype_extent);
      packed_size = max_size;

      // receive data from the right
      recv_tag = collGenerateAlltoallTag(global_rank, right, global_comm);
      res      = MPI_Irecv((char*)recvbuf + rdispls[right] * recvtype_extent,
                      recvcounts[right],
                      recvtype,
                      right_mpi_rank,
                      recv_tag,
                      global_comm->comm,
                      &request);
      assert(res == MPI_SUCCESS);
    }

    if ((left != right) && (0 != recvcounts[left])) {
      // send data to the left
      send_tag = collGenerateAlltoallTag(left, global_rank, global_comm);
      res      = MPI_Send((char*)recvbuf + rdispls[left] * recvtype_extent,
                     recvcounts[left],
                     recvtype,
                     left_mpi_rank,
                     send_tag,
                     global_comm->comm);
      assert(res == MPI_SUCCESS);

      res = MPI_Wait(&request, MPI_STATUSES_IGNORE);
      assert(res == MPI_SUCCESS);

      // receive data from the left
      recv_tag = collGenerateAlltoallTag(global_rank, left, global_comm);
      res      = MPI_Irecv((char*)recvbuf + rdispls[left] * recvtype_extent,
                      recvcounts[left],
                      recvtype,
                      left_mpi_rank,
                      recv_tag,
                      global_comm->comm,
                      &request);
      assert(res == MPI_SUCCESS);
    }

    if (0 != recvcounts[right]) { /* nothing to exchange with the peer on the right */
      // send data to the right
      send_tag = collGenerateAlltoallTag(right, global_rank, global_comm);
      res =
        MPI_Send(tmp_buffer, packed_size, MPI_PACKED, right_mpi_rank, send_tag, global_comm->comm);
      assert(res == MPI_SUCCESS);
    }

    res = MPI_Wait(&request, MPI_STATUSES_IGNORE);
    assert(res == MPI_SUCCESS);
  }

  free(tmp_buffer);

  return CollSuccess;
}

int collAlltoallvMPI(const void* sendbuf,
                     const int sendcounts[],
                     const int sdispls[],
                     CollDataType sendtype,
                     void* recvbuf,
                     const int recvcounts[],
                     const int rdispls[],
                     CollDataType recvtype,
                     CollComm global_comm)
{
  int res;
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_sendtype = collDtypeToMPIDtype(sendtype);
  MPI_Datatype mpi_recvtype = collDtypeToMPIDtype(recvtype);

  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(mpi_sendtype, &lb, &sendtype_extent);
  MPI_Type_get_extent(mpi_recvtype, &lb, &recvtype_extent);

  void* sendbuf_tmp = const_cast<void*>(sendbuf);

  // if (sendbuf == recvbuf) {
  //   return collAlltoallvMPIInplace(recvbuf, recvcounts, rdispls, mpi_recvtype, global_comm);
  // }

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    int total_send_count = sdispls[total_size - 1] + sendcounts[total_size - 1];
    sendbuf_tmp          = collAllocateInlineBuffer(recvbuf, sendtype_extent * total_send_count);
  }

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(sendbuf_tmp) +
                static_cast<ptrdiff_t>(sdispls[sendto_global_rank]) * sendtype_extent;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(rdispls[recvfrom_global_rank]) * recvtype_extent;
    int scount        = sendcounts[sendto_global_rank];
    int rcount        = recvcounts[recvfrom_global_rank];
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = collGenerateAlltoallvTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = collGenerateAlltoallvTag(global_rank, recvfrom_global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "i: %d === global_rank %d, mpi rank %d, send %d to %d, send_tag %d, recv %d from %d, "
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
    res = MPI_Sendrecv(src,
                       scount,
                       mpi_sendtype,
                       sendto_mpi_rank,
                       send_tag,
                       dst,
                       rcount,
                       mpi_recvtype,
                       recvfrom_mpi_rank,
                       recv_tag,
                       global_comm->comm,
                       &status);
    assert(res == MPI_SUCCESS);
  }

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate