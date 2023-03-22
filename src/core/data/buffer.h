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
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Type alias definition for legate::Buffer and utility functions for it
 */

namespace legate {

/**
 * @ingroup data
 * @brief A typed buffer class for intra-task temporary allocations
 *
 * Values in a buffer can be accessed by index expressions with legate::Point objects,
 * or via a raw pointer to the underlying allocation, which can be queried with the `ptr` method.
 *
 * `legate::Buffer` is an alias to
 * [`Legion::DeferredBuffer`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L3509-L3609).
 *
 * Note on using temporary buffers in CUDA tasks:
 *
 * We use Legion `DeferredBuffer`, whose lifetime is not connected with the CUDA stream(s) used to
 * launch kernels. The buffer is allocated immediately at the point when `create_buffer` is called,
 * whereas the kernel that uses it is placed on a stream, and may run at a later point. Normally
 * a `DeferredBuffer` is deallocated automatically by Legion once all the kernels launched in the
 * task are complete. However, a `DeferredBuffer` can also be deallocated immediately using
 * `destroy()`, which is useful for operations that want to deallocate intermediate memory as soon
 * as possible. This deallocation is not synchronized with the task stream, i.e. it may happen
 * before a kernel which uses the buffer has actually completed. This is safe as long as we use the
 * same stream on all GPU tasks running on the same device (which is guaranteed by the current
 * implementation of `get_cached_stream`), because then all the actual uses of the buffer are done
 * in order on the one stream. It is important that all library CUDA code uses
 * `get_cached_stream()`, and all CUDA operations (including library calls) are enqueued on that
 * stream exclusively. This analysis additionally assumes that no code outside of Legate is
 * concurrently allocating from the eager pool, and that it's OK for kernels to access a buffer even
 * after it's technically been deallocated.
 */
template <typename VAL, int32_t DIM = 1>
using Buffer = Legion::DeferredBuffer<VAL, DIM>;

/**
 * @ingroup data
 * @brief Creates a `Buffer` of specific extents
 *
 * @param extents Extents of the buffer
 * @param kind Kind of the target memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A `Buffer` object
 */
template <typename VAL, int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents,
                               Memory::Kind kind = Memory::Kind::NO_MEMKIND,
                               size_t alignment  = 16)
{
  if (Memory::Kind::NO_MEMKIND == kind) kind = find_memory_kind_for_executing_processor(false);
  auto hi = extents - Point<DIM>::ONES();
  // We just avoid creating empty buffers, as they cause all sorts of headaches.
  for (int32_t idx = 0; idx < DIM; ++idx) hi[idx] = std::max<int64_t>(hi[idx], 0);
  Rect<DIM> bounds(Point<DIM>::ZEROES(), hi);
  return Buffer<VAL, DIM>(bounds, kind, nullptr, alignment);
}

/**
 * @ingroup data
 * @brief Creates a `Buffer` of a specific size. Always returns a 1D buffer.
 *
 * @param size Size of the buffdr
 * @param kind Kind of the target memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A 1D `Buffer` object
 */
template <typename VAL>
Buffer<VAL> create_buffer(size_t size,
                          Memory::Kind kind = Memory::Kind::NO_MEMKIND,
                          size_t alignment  = 16)
{
  return create_buffer<VAL, 1>(Point<1>(size), kind, alignment);
}

}  // namespace legate
