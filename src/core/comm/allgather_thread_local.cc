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

int collAllgatherLocal(const void* sendbuf,
                       int sendcount,
                       CollDataType sendtype,
                       void* recvbuf,
                       int recvcount,
                       CollDataType recvtype,
                       CollComm global_comm)
{
  assert(recvcount == sendcount);
  assert(sendtype == recvtype);

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int sendtype_extent = collGetDtypeSize(sendtype);
  int recvtype_extent = collGetDtypeSize(recvtype);

  const void* sendbuf_tmp = sendbuf;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = collAllocateInplaceBuffer(recvbuf, sendtype_extent * sendcount);
  }

  global_comm->comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  for (int recvfrom_global_rank = 0; recvfrom_global_rank < total_size; recvfrom_global_rank++) {
    // wait for other threads to update the buffer address
    while (global_comm->comm->buffers[recvfrom_global_rank] == nullptr)
      ;
    const void* src = global_comm->comm->buffers[recvfrom_global_rank];
    char* dst       = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * recvtype_extent * recvcount;
#ifdef DEBUG_LEGATE
    log_coll.debug("i: %d === global_rank %d, dtype %d, copy rank %d (%p) to rank %d (%p)",
                   i,
                   global_rank,
                   sendtype_extent,
                   recvfrom_global_rank,
                   src,
                   global_rank,
                   dst);
#endif
    memcpy(dst, src, sendcount * sendtype_extent);
  }

  collBarrierLocal(global_comm);
  if (sendbuf == recvbuf) { free(const_cast<void*>(sendbuf_tmp)); }

  __sync_synchronize();

  collUpdateBuffer(global_comm);
  collBarrierLocal(global_comm);

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate