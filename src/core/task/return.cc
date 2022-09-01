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
#ifdef LEGATE_USE_CUDA
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/utilities/typedefs.h"
#endif

using namespace Legion;

namespace legate {

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

static void returned_exception_init(const ReductionOp* reduction_op, void*& ptr, size_t& size)
{
  pack_returned_exception(JoinReturnedException::identity, ptr, size);
}

static void returned_exception_fold(const ReductionOp* reduction_op,
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
  auto buffer      = create_buffer<int8_t>(buffer_size, mem_kind);
  auto p_buffer    = buffer.ptr(0);
  legion_serialize(p_buffer);

  return ReturnValue(p_buffer, buffer_size);
}

ReturnValues::ReturnValues() {}

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values, bool has_exception)
  : has_exception_(has_exception), return_values_(std::move(return_values))
{
  if (return_values_.size() > 1) {
    buffer_size_ += sizeof(uint32_t);
    for (auto& ret : return_values_) buffer_size_ += sizeof(uint32_t) + ret.second;
  } else if (return_values_.size() > 0)
    buffer_size_ = return_values_[0].second;
}

ReturnValue ReturnValues::operator[](int32_t idx) const { return return_values_[idx]; }

size_t ReturnValues::legion_buffer_size() const { return buffer_size_; }

void ReturnValues::legion_serialize(void* buffer) const
{
  auto ptr = static_cast<int8_t*>(buffer);
  if (return_values_.size() > 1) {
    *reinterpret_cast<uint32_t*>(ptr) = return_values_.size();
    ptr += sizeof(uint32_t);
    for (auto& ret : return_values_) {
      *reinterpret_cast<uint32_t*>(ptr) = ret.second;
      ptr += sizeof(uint32_t);
      memcpy(ptr, ret.first, ret.second);
      ptr += ret.second;
    }
  } else {
    assert(return_values_.size() == 1);
    memcpy(ptr, return_values_[0].first, return_values_[0].second);
  }
}

void ReturnValues::legion_deserialize(const void* buffer)
{
  auto mem_kind = find_memory_kind_for_executing_processor();

  auto ptr        = static_cast<const int8_t*>(buffer);
  auto num_values = *reinterpret_cast<const uint32_t*>(ptr);
  ptr += sizeof(uint32_t);
  return_values_.resize(num_values);

  for (auto& ret : return_values_) {
    auto size = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    ret.first  = ptr;
    ret.second = size;
    ptr += size;
  }
  buffer_size_ = ptr - static_cast<const int8_t*>(buffer);
}

void ReturnValues::call_postamble(Legion::Context legion_context) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  }

#ifdef LEGATE_USE_CUDA
  auto kind = Processor::get_executing_processor().kind();
  if (kind == Processor::TOC_PROC) {
    // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
    //        as doing so would require the packing to be chained up with all preceding kernels,
    //        potentially launched with different streams, within the task. Until we find
    //        the right approach, we simply synchronize the device before proceeding.
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t return_size = legion_buffer_size();

    auto mem_kind = (return_values_.size() == 1 && !has_exception_)
                      // The majority of cases would take this code path; there'd be a single
                      // value (of a primitive type) from a task and no exception raised. For those
                      // cases, we allocate the return instance on the framebuffer so that
                      // downstream operations can keep the data within the GPU.
                      ? Legion::Memory::Kind::GPU_FB_MEM
                      // If there were more than one return value or an exception was raised,
                      // we would need to take the slow host-side code path in downstream operations
                      // for demux and/or serdez on packed values. For those cases, keeping the data
                      // in a framebuffer is actually futile, so we just create the return instance
                      // on the zero-copy memory to avoid any unnecessary data movement.
                      : Legion::Memory::Kind::Z_COPY_MEM;

    Rect<1> bounds(0, 0);
    auto return_buffer = Legion::UntypedDeferredBuffer(return_size, 1, mem_kind, bounds);
    AccessorWO<int8_t, 1> acc(return_buffer, bounds, return_size, false);
    void* return_ptr = acc.ptr(0);

    if (return_values_.size() == 1 && !has_exception_) {
      auto stream = cuda::StreamPool::get_stream_pool().get_stream();
      CHECK_CUDA(cudaMemcpyAsync(
        return_ptr, return_values_.front().first, return_size, cudaMemcpyHostToDevice, stream));
    } else
      legion_serialize(return_ptr);

    Legion::Runtime::legion_task_postamble(
      legion_context, return_ptr, return_size, true, return_buffer.get_instance());
    return;
  }
#endif

  if (return_values_.size() == 1) {
    auto& return_value = return_values_.front();
    Legion::Runtime::legion_task_postamble(legion_context, return_value.first, return_value.second);
  } else {
    size_t buffer_size = legion_buffer_size();
    void* buffer       = malloc(buffer_size);
    legion_serialize(buffer);
    Legion::Runtime::legion_task_postamble(legion_context, buffer, buffer_size, true);
  }
}

void register_exception_reduction_op(Runtime* runtime, const LibraryContext& context)
{
  auto redop_id = context.get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  auto* redop   = Realm::ReductionOpUntyped::create_reduction_op<JoinReturnedException>();
  Runtime::register_reduction_op(redop_id, redop, returned_exception_init, returned_exception_fold);
}

}  // namespace legate
