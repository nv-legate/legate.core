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
 
int collBcastMPI(void *buf, int count, collDataType_t type, 
                 int root,
                 collComm_t global_comm)
{	
  int res;

  // int total_size = global_comm.mpi_comm_size * global_comm.nb_threads;
  int total_size = global_comm->global_comm_size;
	MPI_Status status;
 
  int global_rank = global_comm->global_rank;

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  assert(global_comm->starting_tag >= 0);
  
  // non-root
  if (global_rank != root) {
    tag = global_comm->starting_tag * 10000 + global_rank;
#ifdef DEBUG_PRINT
    printf("Bcast Recv global_rank %d, rank %d, tid %d, send to %d (%d), tag %d\n", 
      global_rank, global_comm.mpi_rank, global_comm.tid, 
      root, root_mpi_rank, tag);
#endif
    return MPI_Recv(buf, count, type, root_mpi_rank, tag, global_comm->comm, &status);
  } 

  // root
  int sendto_mpi_rank;
	for(int i = 0 ; i < total_size; i++) {
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = global_comm->starting_tag * 10000 + i;
#ifdef DEBUG_PRINT
    printf("Bcast i: %d === global_rank %d, rank %d, tid %d, send to %d (%d), tag %d\n", 
      i, global_rank, global_comm.mpi_rank, global_comm.tid, 
      i, sendto_mpi_rank, tag);
#endif
    if (global_rank != i) {
      res = MPI_Send(buf, count, type, sendto_mpi_rank, tag, global_comm->comm);
      assert(res == MPI_SUCCESS);
    }
	}

  return collSuccess;
}