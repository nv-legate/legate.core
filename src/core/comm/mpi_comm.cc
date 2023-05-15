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
#include "legate.h"
#include "legion.h"

namespace legate {
namespace comm {
namespace coll {

extern Logger log_coll;

enum CollTag : int {
  BCAST_TAG     = 0,
  GATHER_TAG    = 1,
  ALLTOALL_TAG  = 2,
  ALLTOALLV_TAG = 3,
  MAX_TAG       = 10,
};

static inline std::pair<int, int> mostFrequent(const int* arr, int n);
static inline int match2ranks(int rank1, int rank2, CollComm global_comm);

inline void check_mpi(int error, const char* file, int line)
{
  if (error != MPI_SUCCESS) {
    fprintf(
      stderr, "Internal MPI failure with error code %d in file %s at line %d\n", error, file, line);
#ifdef DEBUG_LEGATE
    assert(false);
#else
    exit(error);
#endif
  }
}

#define CHECK_MPI(expr)                    \
  do {                                     \
    int result = (expr);                   \
    check_mpi(result, __FILE__, __LINE__); \
  } while (false)

// public functions start from here

MPINetwork::MPINetwork(int argc, char* argv[])
  : BackendNetwork(), mpi_tag_ub(0), self_init_mpi(false)
{
  log_coll.debug("Enable MPINetwork");
  assert(current_unique_id == 0);
  int provided, init_flag = 0;
  CHECK_MPI(MPI_Initialized(&init_flag));
  if (!init_flag) {
    log_coll.info("MPI being initialized by legate");
    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
    self_init_mpi = true;
  }
  int mpi_thread_model;
  MPI_Query_thread(&mpi_thread_model);
  if (mpi_thread_model != MPI_THREAD_MULTIPLE) {
    log_coll.fatal(
      "MPI has been initialized by others, but is not initialized with "
      "MPI_THREAD_MULTIPLE");
    LEGATE_ABORT;
  }
  // check
  int *tag_ub, flag;
  CHECK_MPI(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag));
  assert(flag);
  mpi_tag_ub = *tag_ub;
  assert(mpi_comms.empty());
  BackendNetwork::coll_inited = true;
  BackendNetwork::comm_type   = CollCommType::CollMPI;
}

MPINetwork::~MPINetwork()
{
  log_coll.debug("Finalize MPINetwork");
  assert(BackendNetwork::coll_inited == true);
  for (MPI_Comm& mpi_comm : mpi_comms) { CHECK_MPI(MPI_Comm_free(&mpi_comm)); }
  mpi_comms.clear();
  int fina_flag = 0;
  CHECK_MPI(MPI_Finalized(&fina_flag));
  if (fina_flag == 1) {
    log_coll.fatal("MPI should not have been finalized");
    LEGATE_ABORT;
  }
  if (self_init_mpi) {
    MPI_Finalize();
    log_coll.info("finalize mpi");
  }
  BackendNetwork::coll_inited = false;
}

int MPINetwork::init_comm()
{
  int id = 0;
  collGetUniqueId(&id);
#ifdef DEBUG_LEGATE
  int mpi_rank;
  int send_id = id;
  // check if all ranks get the same unique id
  CHECK_MPI(MPI_Bcast(&send_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
  assert(send_id == id);
#endif
  assert(mpi_comms.size() == id);
  // create mpi comm
  MPI_Comm mpi_comm;
  CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
  mpi_comms.push_back(mpi_comm);
  log_coll.debug("Init comm id %d", id);
  return id;
}

int MPINetwork::comm_create(CollComm global_comm,
                            int global_comm_size,
                            int global_rank,
                            int unique_id,
                            const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
  int mpi_rank, mpi_comm_size;
  int *tag_ub, flag;
  int compare_result;
  MPI_Comm comm = mpi_comms[unique_id];
  CHECK_MPI(MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_result));
  assert(MPI_CONGRUENT == compare_result);

  CHECK_MPI(MPI_Comm_rank(comm, &mpi_rank));
  CHECK_MPI(MPI_Comm_size(comm, &mpi_comm_size));
  global_comm->mpi_comm_size = mpi_comm_size;
  global_comm->mpi_rank      = mpi_rank;
  global_comm->mpi_comm      = comm;
  assert(mapping_table != nullptr);
  global_comm->mapping_table.global_rank = (int*)malloc(sizeof(int) * global_comm_size);
  global_comm->mapping_table.mpi_rank    = (int*)malloc(sizeof(int) * global_comm_size);
  memcpy(global_comm->mapping_table.mpi_rank, mapping_table, sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) { global_comm->mapping_table.global_rank[i] = i; }
  std::pair<int, int> p             = mostFrequent(mapping_table, global_comm_size);
  global_comm->nb_threads           = p.first;
  global_comm->mpi_comm_size_actual = p.second;
  return CollSuccess;
}

int MPINetwork::comm_destroy(CollComm global_comm)
{
  if (global_comm->mapping_table.global_rank != nullptr) {
    free(global_comm->mapping_table.global_rank);
    global_comm->mapping_table.global_rank = nullptr;
  }
  if (global_comm->mapping_table.mpi_rank != nullptr) {
    free(global_comm->mapping_table.mpi_rank);
    global_comm->mapping_table.mpi_rank = nullptr;
  }
  global_comm->status = false;
  return CollSuccess;
}

int MPINetwork::alltoallv(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          CollDataType type,
                          CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(const_cast<void*>(sendbuf)) +
                static_cast<ptrdiff_t>(sdispls[sendto_global_rank]) * type_extent;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
    int scount        = sendcounts[sendto_global_rank];
    int rcount        = recvcounts[recvfrom_global_rank];
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = generateAlltoallvTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = generateAlltoallvTag(global_rank, recvfrom_global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallvMPI i: %d === global_rank %d, mpi rank %d, send to %d (%d), send_tag %d, "
      "recv from %d (%d), "
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
                           scount,
                           mpi_type,
                           sendto_mpi_rank,
                           send_tag,
                           dst,
                           rcount,
                           mpi_type,
                           recvfrom_mpi_rank,
                           recv_tag,
                           global_comm->mpi_comm,
                           &status));
  }

  return CollSuccess;
}

int MPINetwork::alltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(const_cast<void*>(sendbuf)) +
                static_cast<ptrdiff_t>(sendto_global_rank) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    assert(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    assert(recvfrom_global_rank == global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    int send_tag = generateAlltoallTag(sendto_global_rank, global_rank, global_comm);
    int recv_tag = generateAlltoallTag(global_rank, recvfrom_global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "AlltoallMPI i: %d === global_rank %d, mpi rank %d, send to %d (%d), send_tag %d, "
      "recv from %d (%d), "
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
                           global_comm->mpi_comm,
                           &status));
  }

  return CollSuccess;
}

int MPINetwork::allgather(
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

  gather(sendbuf_tmp, recvbuf, count, type, 0, global_comm);

  bcast(recvbuf, count * total_size, type, 0, global_comm);

  if (sendbuf == recvbuf) { free(sendbuf_tmp); }

  return CollSuccess;
}

int MPINetwork::gather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm)
{
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // Should not see inplace here
  if (sendbuf == recvbuf) { assert(0); }

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  // non-root
  if (global_rank != root) {
    tag = generateGatherTag(global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("GatherMPI: non-root send global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   global_rank,
                   global_comm->mpi_rank,
                   root,
                   root_mpi_rank,
                   tag);
#endif
    CHECK_MPI(MPI_Send(sendbuf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm));
    return CollSuccess;
  }

  // root
  MPI_Aint incr, lb, type_extent;
  MPI_Type_get_extent(mpi_type, &lb, &type_extent);
  incr      = type_extent * static_cast<ptrdiff_t>(count);
  char* dst = static_cast<char*>(recvbuf);
  int recvfrom_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = generateGatherTag(i, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug(
      "GatherMPI: root i %d === global_rank %d, mpi rank %d, recv %p, from %d (%d), tag %d",
      i,
      global_rank,
      global_comm->mpi_rank,
      dst,
      i,
      recvfrom_mpi_rank,
      tag);
#endif
    assert(dst != nullptr);
    if (global_rank == i) {
      memcpy(dst, sendbuf, incr);
    } else {
      CHECK_MPI(
        MPI_Recv(dst, count, mpi_type, recvfrom_mpi_rank, tag, global_comm->mpi_comm, &status));
    }
    dst += incr;
  }

  return CollSuccess;
}

int MPINetwork::bcast(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  int tag;
  MPI_Status status;

  int total_size  = global_comm->global_comm_size;
  int global_rank = global_comm->global_rank;

  int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  assert(root == global_comm->mapping_table.global_rank[root]);

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // non-root
  if (global_rank != root) {
    tag = generateBcastTag(global_rank, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("BcastMPI: non-root recv global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   global_rank,
                   global_comm->mpi_rank,
                   root,
                   root_mpi_rank,
                   tag);
#endif
    CHECK_MPI(MPI_Recv(buf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm, &status));
    return CollSuccess;
  }

  // root
  int sendto_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    assert(i == global_comm->mapping_table.global_rank[i]);
    tag = generateBcastTag(i, global_comm);
#ifdef DEBUG_LEGATE
    log_coll.debug("BcastMPI: root i %d === global_rank %d, mpi rank %d, send to %d (%d), tag %d",
                   i,
                   global_rank,
                   global_comm->mpi_rank,
                   i,
                   sendto_mpi_rank,
                   tag);
#endif
    if (global_rank != i) {
      CHECK_MPI(MPI_Send(buf, count, mpi_type, sendto_mpi_rank, tag, global_comm->mpi_comm));
    }
  }

  return CollSuccess;
}

static inline std::pair<int, int> mostFrequent(const int* arr, int n)
{
  std::unordered_map<int, int> hash;
  for (int i = 0; i < n; i++) hash[arr[i]]++;

  // find the max frequency
  int max_count = 0;
  std::unordered_map<int, int>::iterator it;
  for (it = hash.begin(); it != hash.end(); it++) {
    if (max_count < it->second) { max_count = it->second; }
  }

  return std::make_pair(max_count, hash.size());
}

static inline int match2ranks(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // send_tag = sendto_global_rank * 10000 + global_rank (concat 2 ranks)
  // which dst seg it sends to (in dst rank)
  // recv_tag = global_rank * 10000 + recvfrom_global_rank (concat 2 ranks)
  // idx of current seg we are receving (in src/my rank)
  // example:
  // 00 | 01 | 02 | 03
  // 10 | 11 | 12 | 13
  // 20 | 21 | 22 | 23
  // 30 | 31 | 32 | 33
  // 01's send_tag = 10, 10's recv_tag = 10, match
  // 12's send_tag = 21, 21's recv_tag = 21, match

  int tag;
  // old tagging system for debug
  // constexpr int const max_ranks = 10000;
  // tag                           = rank1 * max_ranks + rank2;

  // new tagging system, if crash, switch to the old one

  tag = rank1 % global_comm->nb_threads * global_comm->global_comm_size + rank2;

  // Szudzik's Function, two numbers < 32768
  // if (rank1 >= rank2) {
  //   tag = rank1*rank1 + rank1 + rank2;
  // } else {
  //   tag = rank1 + rank2*rank2;
  // }

  // Cantor Pairing Function, two numbers < 32768
  // tag = (rank1 + rank2) * (rank1 + rank2 + 1) / 2 + rank1;

  return tag;
}

// protected functions start from here

MPI_Datatype MPINetwork::dtypeToMPIDtype(CollDataType dtype)
{
  switch (dtype) {
    case CollDataType::CollInt8: {
      return MPI_INT8_T;
    }
    case CollDataType::CollChar: {
      return MPI_CHAR;
    }
    case CollDataType::CollUint8: {
      return MPI_UINT8_T;
    }
    case CollDataType::CollInt: {
      return MPI_INT;
    }
    case CollDataType::CollUint32: {
      return MPI_UINT32_T;
    }
    case CollDataType::CollInt64: {
      return MPI_INT64_T;
    }
    case CollDataType::CollUint64: {
      return MPI_UINT64_T;
    }
    case CollDataType::CollFloat: {
      return MPI_FLOAT;
    }
    case CollDataType::CollDouble: {
      return MPI_DOUBLE;
    }
    default: {
      log_coll.fatal("Unknown datatype");
      LEGATE_ABORT;
      return MPI_BYTE;
    }
  }
}

int MPINetwork::generateAlltoallTag(int rank1, int rank2, CollComm global_comm)
{
  int tag = match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int MPINetwork::generateAlltoallvTag(int rank1, int rank2, CollComm global_comm)
{
  int tag = match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int MPINetwork::generateBcastTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
  assert(tag <= mpi_tag_ub && tag >= 0);
  return tag;
}

int MPINetwork::generateGatherTag(int rank, CollComm global_comm)
{
  int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
  assert(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

}  // namespace coll
}  // namespace comm
}  // namespace legate
