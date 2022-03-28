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

namespace legate {

class ScopedAllocator {
 public:
  using ByteBuffer = Buffer<int8_t>;

 public:
  ScopedAllocator() = default;

  // Iff 'scoped', all allocations will be released upon destruction.
  // Otherwise this is up to the runtime after the task has finished.
  ScopedAllocator(Legion::Memory::Kind kind, bool scoped = true, size_t alignment = 16);
  ~ScopedAllocator();

 public:
  void* allocate(size_t bytes);
  void deallocate(void* ptr);

 private:
  Legion::Memory::Kind target_kind_{Legion::Memory::Kind::SYSTEM_MEM};
  bool scoped_;
  size_t alignment_;
  std::unordered_map<const void*, ByteBuffer> buffers_{};
};

}  // namespace legate