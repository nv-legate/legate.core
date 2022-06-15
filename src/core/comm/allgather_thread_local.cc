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

int allgatherLocal(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  const void* sendbuf_tmp = sendbuf;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) { sendbuf_tmp = allocateInplaceBuffer(recvbuf, type_extent * count); }

  global_comm->comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  for (int recvfrom_global_rank = 0; recvfrom_global_rank < total_size; recvfrom_global_rank++) {
    // wait for other threads to update the buffer address
    while (global_comm->comm->buffers[recvfrom_global_rank] == nullptr)
      ;
    const void* src = global_comm->comm->buffers[recvfrom_global_rank];
    char* dst       = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AllgatherLocal i: %d === global_rank %d, dtype %d, copy rank %d (%p) to rank %d (%p)",
      recvfrom_global_rank,
      global_rank,
      type_extent,
      recvfrom_global_rank,
      src,
      global_rank,
      dst);
#endif
    memcpy(dst, src, count * type_extent);
  }

  barrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(const_cast<void*>(sendbuf_tmp)); }

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate