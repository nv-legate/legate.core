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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "coll.h"
 
int collAlltoallvMPI(const void *sendbuf, const int sendcounts[],
                     const int sdispls[], collDataType_t sendtype,
                     void *recvbuf, const int recvcounts[],
                     const int rdispls[], collDataType_t recvtype, 
                     collComm_t global_comm)
{	
  int res;

  // int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  int total_size = global_comm->global_comm_size;
	MPI_Status status;

  MPI_Aint lb, sendtype_extent, recvtype_extent;
  MPI_Type_get_extent(sendtype, &lb, &sendtype_extent);
  MPI_Type_get_extent(recvtype, &lb, &recvtype_extent);

  int global_rank = global_comm->global_rank;

  void *sendbuf_tmp = NULL;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    int total_send_count = sdispls[total_size-1] + sendcounts[total_size-1];
    sendbuf_tmp = (void *)malloc(sendtype_extent * total_send_count);
    memcpy(sendbuf_tmp, recvbuf, sendtype_extent * total_send_count);
    // int * sendval = (int*)sendbuf_tmp;
    // printf("malloc %p, size %ld, [%d]\n", sendbuf_tmp, total_size * recvtype_extent * recvcount, sendval[0]);
  } else {
    sendbuf_tmp = const_cast<void*>(sendbuf);
  }

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
	for(int i = 1 ; i < total_size + 1; i++) {
    sendto_global_rank  = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char *src = (char*)sendbuf_tmp + (ptrdiff_t)sdispls[sendto_global_rank] * sendtype_extent;
    char *dst = (char*)recvbuf + (ptrdiff_t)rdispls[recvfrom_global_rank] * recvtype_extent;
    int scount = sendcounts[sendto_global_rank];
    int rcount = recvcounts[recvfrom_global_rank];
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = (sendto_global_rank * 10000 + global_rank) * 10 + ALLTOALLV_TAG; // which dst seg it sends to (in dst rank)
    int recv_tag = (global_rank * 10000 + recvfrom_global_rank) * 10 + ALLTOALLV_TAG; // idx of current seg we are receving (in src/my rank)
#ifdef DEBUG_PRINT
    printf("i: %d === global_rank %d, mpi rank %d, send %d to %d, send_tag %d, recv %d from %d, recv_tag %d\n", 
      i, global_rank, global_comm->mpi_rank, 
      sendto_global_rank, sendto_mpi_rank, send_tag, recvfrom_global_rank, recvfrom_mpi_rank, recv_tag);
#endif
    res = MPI_Sendrecv(src, scount, sendtype, sendto_mpi_rank, send_tag, dst, rcount, recvtype, recvfrom_mpi_rank, recv_tag, global_comm->comm, &status);
    assert(res == MPI_SUCCESS);
	}

  if (sendbuf == recvbuf) {
    free(sendbuf_tmp);
  }

  return collSuccess;
}