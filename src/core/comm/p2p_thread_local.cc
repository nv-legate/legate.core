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

enum P2PTag : int {
  INIT                = 0,
  SEND_BUFFER_READY   = 1,
  SEND_CP_DONE        = 2,
  SEND_BUFFER_RELEASE = 3,
};

int sendLocal(
  const void* sendbuf, int count, CollDataType type, int dest, int tag, CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  int key                                 = global_rank * total_size + dest;
  global_comm->comm->buffers[global_rank] = sendbuf;
  global_comm->comm->buffer_ready[key]    = P2PTag::SEND_BUFFER_READY;
  __sync_synchronize();

  // wait for dest to copy
  while (global_comm->comm->buffer_ready[key] != P2PTag::SEND_CP_DONE)
    ;
  __sync_synchronize();

  // remote thread done with the copy, let reset buffer
  resetLocalBuffer(global_comm);
  global_comm->comm->buffer_ready[key] = P2PTag::SEND_BUFFER_RELEASE;
  __sync_synchronize();

  // wait for dest to reset flag to init
  while (global_comm->comm->buffer_ready[key] != P2PTag::INIT)
    ;

  return CollSuccess;
}

int recvLocal(
  void* recvbuf, int count, CollDataType type, int source, int tag, CollComm global_comm)
{
  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int type_extent = getDtypeSize(type);

  // wait for source to put the buffer
  int key = source * total_size + global_rank;
  while (global_comm->comm->buffer_ready[key] == P2PTag::INIT ||
         global_comm->comm->buffers[source] == nullptr)
    ;
  __sync_synchronize();

  // start memcpy
  memcpy(recvbuf, global_comm->comm->buffers[source], count * type_extent);
  __sync_synchronize();
  global_comm->comm->buffer_ready[key] = P2PTag::SEND_CP_DONE;

  // wait for source to reset the buffer
  while (global_comm->comm->buffer_ready[key] != P2PTag::SEND_BUFFER_RELEASE ||
         global_comm->comm->buffers[source] != nullptr)
    ;
  __sync_synchronize();

  global_comm->comm->buffer_ready[key] = P2PTag::INIT;

  return CollSuccess;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate