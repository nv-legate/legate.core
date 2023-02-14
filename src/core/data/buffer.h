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

#pragma once

#include "legion.h"

#include "core/utilities/machine.h"

namespace legate {

template <typename VAL, int32_t DIM = 1>
using Buffer = Legion::DeferredBuffer<VAL, DIM>;

// Note on using temporary buffers in CUDA tasks:
// We use Legion `DeferredBuffer`s, whose lifetime is not connected with the CUDA stream(s) used to
// launch kernels. The buffer is allocated immediately at the point when `create_buffer` called,
// whereas the kernel that uses it is placed on a stream, and may run at a later point. Normally
// `DeferredBuffer`s are deallocated automatically by Legion once all the kernels launched in the
// task are complete. However, `DeferredBuffer`s can also be deallocated immediately using
// `destroy()`, which is useful for operations that want to deallocate intermediate memory as soon
// as possible. This deallocation is not synchronized with the task stream, i.e. it may happen
// before a kernel which uses the buffer has actually completed. This is safe as long as we use the
// same stream on all GPU tasks running on the same device (which is guaranteed by the current
// implementation of `get_cached_stream()`), because then all the actual uses of the buffer are done
// in order on the one stream. It is important that all library CUDA code uses
// `get_cached_stream()`, and all CUDA operations (including library calls) are enqueued on that
// stream exclusively. This analysis additionally assumes that no code outside of Legate is
// concurrently allocating from the eager pool, and that it's OK for kernels to access a buffer even
// after it's technically been deallocated.

template <typename VAL, int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Legion::Point<DIM>& extents,
                               Legion::Memory::Kind kind = Legion::Memory::Kind::NO_MEMKIND,
                               size_t alignment          = 16)
{
  using namespace Legion;
  if (Memory::Kind::NO_MEMKIND == kind) kind = find_memory_kind_for_executing_processor(false);
  auto hi = extents - Point<DIM>::ONES();
  // We just avoid creating empty buffers, as they cause all sorts of headaches.
  for (int32_t idx = 0; idx < DIM; ++idx) hi[idx] = std::max<int64_t>(hi[idx], 0);
  Rect<DIM> bounds(Point<DIM>::ZEROES(), hi);
  return Buffer<VAL, DIM>(bounds, kind, nullptr, alignment);
}

template <typename VAL>
Buffer<VAL> create_buffer(size_t size,
                          Legion::Memory::Kind kind = Legion::Memory::Kind::NO_MEMKIND,
                          size_t alignment          = 16)
{
  return create_buffer<VAL, 1>(Legion::Point<1>(size), kind, alignment);
}

}  // namespace legate
