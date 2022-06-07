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

int alltoallvLocal(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   CollDataType type,
                   CollComm global_comm)
{
  int res;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  global_comm->comm->displs[global_rank]  = sdispls;
  global_comm->comm->buffers[global_rank] = sendbuf;
  __sync_synchronize();

  int recvfrom_global_rank;
  int recvfrom_seg_id  = global_rank;
  const void* src_base = nullptr;
  const int* displs    = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->comm->buffers[recvfrom_global_rank] == nullptr ||
           global_comm->comm->displs[recvfrom_global_rank] == nullptr)
      ;
    src_base  = global_comm->comm->buffers[recvfrom_global_rank];
    displs    = global_comm->comm->displs[recvfrom_global_rank];
    char* src = static_cast<char*>(const_cast<void*>(src_base)) +
                static_cast<ptrdiff_t>(displs[recvfrom_seg_id]) * type_extent;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallvLocal i: %d === global_rank %d, dtype %d, copy rank %d (seg %d, sdispls %d, %p) to "
      "rank %d (seg "
      "%d, rdispls %d, %p)",
      i,
      global_rank,
      type_extent,
      recvfrom_global_rank,
      recvfrom_seg_id,
      sdispls[recvfrom_seg_id],
      src,
      global_rank,
      recvfrom_global_rank,
      rdispls[recvfrom_global_rank],
      dst);
#endif
    memcpy(dst, src, recvcounts[recvfrom_global_rank] * type_extent);
  }

  barrierLocal(global_comm);

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate