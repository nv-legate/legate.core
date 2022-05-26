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

#define ALLTOALL_USE_SENDRECV

static int collAlltoallMPIInplace(void* recvbuf,
                                  int recvcount,
                                  MPI_Datatype recvtype,
                                  CollComm global_comm)
{
  int res;

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
    res      = MPI_Irecv((char*)recvbuf + right * recvcount * recvtype_extent,
                    recvcount,
                    recvtype,
                    right_mpi_rank,
                    recv_tag,
                    global_comm->comm,
                    &request);
    assert(res == MPI_SUCCESS);

    if (left != right) {
      // send data to the left
      send_tag = collGenerateAlltoallTag(left, global_rank, global_comm);
      res      = MPI_Send((char*)recvbuf + left * recvcount * recvtype_extent,
                     recvcount,
                     recvtype,
                     left_mpi_rank,
                     send_tag,
                     global_comm->comm);
      assert(res == MPI_SUCCESS);

      res = MPI_Wait(&request, MPI_STATUSES_IGNORE);
      assert(res == MPI_SUCCESS);

      // receive data from the left
      recv_tag = collGenerateAlltoallTag(global_rank, left, global_comm);
      res      = MPI_Irecv((char*)recvbuf + left * recvcount * recvtype_extent,
                      recvcount,
                      recvtype,
                      left_mpi_rank,
                      recv_tag,
                      global_comm->comm,
                      &request);
      assert(res == MPI_SUCCESS);
    }

    // send data to the right
    assert(packed_size == recvtype_extent * recvcount);
    send_tag = collGenerateAlltoallTag(right, global_rank, global_comm);
    res =
      MPI_Send(tmp_buffer, packed_size, MPI_PACKED, right_mpi_rank, send_tag, global_comm->comm);
    assert(res == MPI_SUCCESS);

    res = MPI_Wait(&request, MPI_STATUSES_IGNORE);
    assert(res == MPI_SUCCESS);
  }

  free(tmp_buffer);

  return CollSuccess;
}

int collAlltoallMPI(const void* sendbuf,
                    int sendcount,
                    CollDataType sendtype,
                    void* recvbuf,
                    int recvcount,
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

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    // Not sure which way is better
    // return collAlltoallMPIInplace(recvbuf, recvcount, mpi_recvtype, global_comm);
    sendbuf_tmp = collAllocateInplaceBuffer(recvbuf, total_size * sendtype_extent * sendcount);
  }

#ifdef ALLTOALL_USE_SENDRECV
  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(sendbuf_tmp) +
                static_cast<ptrdiff_t>(sendto_global_rank) * sendtype_extent * sendcount;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * recvtype_extent * recvcount;
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = collGenerateAlltoallTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = collGenerateAlltoallTag(global_rank, recvfrom_global_rank, global_comm);
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
                       sendcount,
                       mpi_sendtype,
                       sendto_mpi_rank,
                       send_tag,
                       dst,
                       recvcount,
                       mpi_recvtype,
                       recvfrom_mpi_rank,
                       recv_tag,
                       global_comm->comm,
                       &status);
    assert(res == MPI_SUCCESS);
  }
#elif defined(ALLTOALL_USE_SENDRECV_OLD)
  int dest_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    char* src     = (char*)sendbuf_tmp + i * sendtype_extent * sendcount;
    char* dst     = (char*)recvbuf + i * recvtype_extent * recvcount;
    dest_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    int send_tag  = collGenerateAlltoallTag(i, global_rank, global_comm);
    int recv_tag  = collGenerateAlltoallTag(global_rank, i, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "i: %d === global_rank %d, mpi rank %d, send %d to %d, send_tag %d, recv %d from %d, "
      "recv_tag %d",
      i,
      global_rank,
      global_comm->mpi_rank,
      i,
      dest_mpi_rank,
      send_tag,
      i,
      dest_mpi_rank,
      recv_tag);
#endif
    res = MPI_Sendrecv(src,
                       sendcount,
                       mpi_sendtype,
                       dest_mpi_rank,
                       send_tag,
                       dst,
                       recvcount,
                       mpi_recvtype,
                       dest_mpi_rank,
                       recv_tag,
                       global_comm->comm,
                       &status);
    assert(res == MPI_SUCCESS);
  }
#endif

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate