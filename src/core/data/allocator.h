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

#pragma once

#include "core/data/buffer.h"

#include <unordered_map>

/**
 * @file
 * @brief Class definition for legate::ScopedAllocator
 */

namespace legate {

/**
 * @ingroup data
 * @brief A simple allocator backed by `Buffer` objects
 *
 * For each allocation request, this allocator creates a 1D `Buffer` of `int8_t` and returns
 * the raw pointer to it. By default, all allocations are deallocated when the allocator is
 * destroyed, and can optionally be made alive until the task finishes by making the allocator
 * unscoped.
 */
class ScopedAllocator {
 public:
  using ByteBuffer = Buffer<int8_t>;

 public:
  ScopedAllocator() = default;

  // Iff 'scoped', all allocations will be released upon destruction.
  // Otherwise this is up to the runtime after the task has finished.
  /**
   * @brief Create a `ScopedAllocator` for a specific memory kind
   *
   * @param kind Kind of the memory on which the `Buffer`s should be created
   * @param scoped If true, the allocator is scoped; i.e., lifetimes of allocations are tied to
   * the allocator's lifetime. Otherwise, the allocations are alive until the task finishes
   * (and unless explicitly deallocated).
   * @param alignment Alignment for the allocations
   */
  ScopedAllocator(Memory::Kind kind, bool scoped = true, size_t alignment = 16);
  ~ScopedAllocator();

 public:
  /**
   * @brief Allocates a contiguous buffer of the given Memory::Kind
   *
   * When the allocator runs out of memory, the runtime will fail with an error message.
   * Otherwise, the function returns a valid pointer.
   *
   * @param bytes Size of the allocation in bytes
   *
   * @return A raw pointer to the allocation
   */
  void* allocate(size_t bytes);
  /**
   * @brief Deallocates an allocation. The input pointer must be one that was previously
   * returned by an `allocate` call, otherwise the code will fail with an error message.
   *
   * @param ptr Pointer to the allocation to deallocate
   */
  void deallocate(void* ptr);

 private:
  Memory::Kind target_kind_{Memory::Kind::SYSTEM_MEM};
  bool scoped_;
  size_t alignment_;
  std::unordered_map<const void*, ByteBuffer> buffers_{};
};

}  // namespace legate
