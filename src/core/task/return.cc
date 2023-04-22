/* Copyright 2021-2022 NVIDIA Corporation
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
#include <stdint.h>
#include <string.h>

#include "legion.h"

#include "core/data/buffer.h"
#include "core/legate_c.h"
#include "core/runtime/context.h"
#include "core/task/return.h"
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"
#ifdef LEGATE_USE_CUDA
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#endif

namespace legate {

ReturnValue::ReturnValue(Legion::UntypedDeferredValue value, size_t size)
  : value_(value), size_(size)
{
  is_device_value_ = value.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM;
}

/*static*/ ReturnValue ReturnValue::unpack(const void* ptr, size_t size, Memory::Kind memory_kind)
{
  ReturnValue result(Legion::UntypedDeferredValue(size, memory_kind), size);
#ifdef DEBUG_LEGATE
  assert(!result.is_device_value());
#endif
  memcpy(result.ptr(), ptr, size);

  return result;
}

void ReturnValue::finalize(Legion::Context legion_context) const
{
  value_.finalize(legion_context);
}

void* ReturnValue::ptr()
{
  AccessorRW<int8_t, 1> acc(value_, size_, false);
  return acc.ptr(0);
}

const void* ReturnValue::ptr() const
{
  AccessorRO<int8_t, 1> acc(value_, size_, false);
  return acc.ptr(0);
}

struct JoinReturnedException {
  using LHS = ReturnedException;
  using RHS = LHS;

  static const ReturnedException identity;

  template <bool EXCLUSIVE>
  static void apply(LHS& lhs, RHS rhs)
  {
#ifdef DEBUG_LEGATE
    assert(EXCLUSIVE);
#endif
    if (lhs.raised() || !rhs.raised()) return;
    lhs = rhs;
  }

  template <bool EXCLUSIVE>
  static void fold(RHS& rhs1, RHS rhs2)
  {
#ifdef DEBUG_LEGATE
    assert(EXCLUSIVE);
#endif
    if (rhs1.raised() || !rhs2.raised()) return;
    rhs1 = rhs2;
  }
};

/*static*/ const ReturnedException JoinReturnedException::identity;

static void pack_returned_exception(const ReturnedException& value, void*& ptr, size_t& size)
{
  auto new_size = value.legion_buffer_size();
  if (new_size > size) {
    size = new_size;
    ptr  = realloc(ptr, new_size);
  }
  value.legion_serialize(ptr);
}

static void returned_exception_init(const Legion::ReductionOp* reduction_op,
                                    void*& ptr,
                                    size_t& size)
{
  pack_returned_exception(JoinReturnedException::identity, ptr, size);
}

static void returned_exception_fold(const Legion::ReductionOp* reduction_op,
                                    void*& lhs_ptr,
                                    size_t& lhs_size,
                                    const void* rhs_ptr)

{
  ReturnedException lhs, rhs;
  lhs.legion_deserialize(lhs_ptr);
  rhs.legion_deserialize(rhs_ptr);
  JoinReturnedException::fold<true>(lhs, rhs);
  pack_returned_exception(lhs, lhs_ptr, lhs_size);
}

ReturnedException::ReturnedException(int32_t index, const std::string& error_message)
  : raised_(true), index_(index), error_message_(error_message)
{
}

size_t ReturnedException::legion_buffer_size() const
{
  size_t size = sizeof(bool);
  if (raised_) size += sizeof(int32_t) + sizeof(uint32_t) + error_message_.size();
  return size;
}

void ReturnedException::legion_serialize(void* buffer) const
{
  int8_t* ptr                   = static_cast<int8_t*>(buffer);
  *reinterpret_cast<bool*>(ptr) = raised_;
  if (raised_) {
    ptr += sizeof(bool);
    *reinterpret_cast<int32_t*>(ptr) = index_;
    ptr += sizeof(int32_t);
    uint32_t error_len                = static_cast<uint32_t>(error_message_.size());
    *reinterpret_cast<uint32_t*>(ptr) = error_len;
    ptr += sizeof(uint32_t);
    memcpy(ptr, error_message_.c_str(), error_len);
  }
}

void ReturnedException::legion_deserialize(const void* buffer)
{
  const int8_t* ptr = static_cast<const int8_t*>(buffer);
  raised_           = *reinterpret_cast<const bool*>(ptr);
  if (raised_) {
    ptr += sizeof(bool);
    index_ = *reinterpret_cast<const int32_t*>(ptr);
    ptr += sizeof(int32_t);
    uint32_t error_len = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    error_message_ = std::string(ptr, ptr + error_len);
  }
}

ReturnValue ReturnedException::pack() const
{
  auto buffer_size = legion_buffer_size();
  auto mem_kind    = find_memory_kind_for_executing_processor();
  auto buffer      = Legion::UntypedDeferredValue(buffer_size, mem_kind);

  AccessorWO<int8_t, 1> acc(buffer, buffer_size, false);
  legion_serialize(acc.ptr(0));

  return ReturnValue(buffer, buffer_size);
}

ReturnValues::ReturnValues() {}

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values)
  : return_values_(std::move(return_values))
{
  if (return_values_.size() > 1) {
    buffer_size_ += sizeof(uint32_t);
    for (auto& ret : return_values_) buffer_size_ += sizeof(uint32_t) + ret.size();
  } else if (return_values_.size() > 0)
    buffer_size_ = return_values_[0].size();
}

ReturnValue ReturnValues::operator[](int32_t idx) const { return return_values_[idx]; }

size_t ReturnValues::legion_buffer_size() const { return buffer_size_; }

void ReturnValues::legion_serialize(void* buffer) const
{
  // We pack N return values into the buffer in the following format:
  //
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  // |   #    | offset to |     | offset to  | total | value | value | ...
  // | values | scalar 1  | ... | scalar N-1 | value |   1   |   2   |
  // |        |           |     |            | size  |       |       |
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  //           <============ offsets ===============> <==== values =======>
  //
  // the size of value i is computed by offsets[i] - (i == 0 ? 0 : offsets[i-1])

  // Special case with a single scalar
  if (return_values_.size() == 1) {
    auto& ret = return_values_.front();
#ifdef LEGATE_USE_CUDA
    if (ret.is_device_value()) {
#ifdef DEBUG_LEGATE
      assert(Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC);
#endif
      CHECK_CUDA(cudaMemcpyAsync(buffer,
                                 ret.ptr(),
                                 ret.size(),
                                 cudaMemcpyDeviceToHost,
                                 cuda::StreamPool::get_stream_pool().get_stream()));
    } else
#endif
      memcpy(buffer, ret.ptr(), ret.size());
    return;
  }

  *static_cast<uint32_t*>(buffer) = return_values_.size();
  auto ptr                        = static_cast<int8_t*>(buffer) + sizeof(uint32_t);

  uint32_t offset = 0;
  for (auto ret : return_values_) {
    offset += ret.size();
    *reinterpret_cast<uint32_t*>(ptr) = offset;
    ptr                               = ptr + sizeof(uint32_t);
  }

#ifdef LEGATE_USE_CUDA
  if (Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();
    for (auto ret : return_values_) {
      uint32_t size = ret.size();
      if (ret.is_device_value())
        CHECK_CUDA(cudaMemcpyAsync(ptr, ret.ptr(), size, cudaMemcpyDeviceToHost, stream));
      else
        memcpy(ptr, ret.ptr(), size);
      ptr += size;
    }
  } else
#endif
  {
    for (auto ret : return_values_) {
      uint32_t size = ret.size();
      memcpy(ptr, ret.ptr(), size);
      ptr += size;
    }
  }
}

void ReturnValues::legion_deserialize(const void* buffer)
{
  auto mem_kind = find_memory_kind_for_executing_processor();

  auto ptr        = static_cast<const int8_t*>(buffer);
  auto num_values = *reinterpret_cast<const uint32_t*>(ptr);

  auto offsets = reinterpret_cast<const uint32_t*>(ptr + sizeof(uint32_t));
  auto values  = ptr + sizeof(uint32_t) + sizeof(uint32_t) * num_values;

  uint32_t offset = 0;
  for (uint32_t idx = 0; idx < num_values; ++idx) {
    uint32_t next_offset = offsets[idx];
    uint32_t size        = next_offset - offset;
    return_values_.push_back(ReturnValue::unpack(values + offset, size, mem_kind));
    offset = next_offset;
  }
}

/*static*/ ReturnValue ReturnValues::extract(Legion::Future future, uint32_t to_extract)
{
  auto kind          = find_memory_kind_for_executing_processor();
  const auto* buffer = future.get_buffer(kind);

  auto ptr        = static_cast<const int8_t*>(buffer);
  auto num_values = *reinterpret_cast<const uint32_t*>(ptr);

  auto offsets = reinterpret_cast<const uint32_t*>(ptr + sizeof(uint32_t));
  auto values  = ptr + sizeof(uint32_t) + sizeof(uint32_t) * num_values;

  uint32_t next_offset = offsets[to_extract];
  uint32_t offset      = to_extract == 0 ? 0 : offsets[to_extract - 1];
  uint32_t size        = next_offset - offset;

  return ReturnValue::unpack(values + offset, size, kind);
}

void ReturnValues::finalize(Legion::Context legion_context) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  } else if (return_values_.size() == 1) {
    return_values_.front().finalize(legion_context);
    return;
  }

#ifdef LEGATE_USE_CUDA
  auto kind = Processor::get_executing_processor().kind();
  // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
  //        as doing so would require the packing to be chained up with all preceding kernels,
  //        potentially launched with different streams, within the task. Until we find
  //        the right approach, we simply synchronize the device before proceeding.
  if (kind == Processor::TOC_PROC) CHECK_CUDA(cudaDeviceSynchronize());
#endif

  size_t return_size = legion_buffer_size();
  auto return_buffer =
    Legion::UntypedDeferredValue(return_size, find_memory_kind_for_executing_processor());
  AccessorWO<int8_t, 1> acc(return_buffer, return_size, false);
  legion_serialize(acc.ptr(0));
  return_buffer.finalize(legion_context);
}

void register_exception_reduction_op(Legion::Runtime* runtime, const LibraryContext* context)
{
  auto redop_id = context->get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  auto* redop   = Realm::ReductionOpUntyped::create_reduction_op<JoinReturnedException>();
  Legion::Runtime::register_reduction_op(
    redop_id, redop, returned_exception_init, returned_exception_fold);
}

}  // namespace legate
