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

class DeferredBufferAllocator {
 public:
  using ByteBuffer = Buffer<int8_t>;

 public:
  DeferredBufferAllocator() = default;
  DeferredBufferAllocator(Legion::Memory::Kind kind);
  virtual ~DeferredBufferAllocator();

 public:
  typedef char value_type;
  char* allocate(size_t bytes);
  void deallocate(char* ptr, size_t n);

 private:
  Legion::Memory::Kind target_kind{Legion::Memory::Kind::SYSTEM_MEM};
  std::unordered_map<const void*, ByteBuffer> buffers{};
};

}  // namespace legate