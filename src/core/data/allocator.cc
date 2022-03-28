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

#include "core/data/allocator.h"
#include "legate.h"

namespace legate {

ScopedAllocator::ScopedAllocator(Legion::Memory::Kind kind, bool scoped, size_t alignment)
  : target_kind_(kind), scoped_(scoped), alignment_(alignment)
{
}

ScopedAllocator::~ScopedAllocator()
{
  if (scoped_) {
    for (auto& pair : buffers_) { pair.second.destroy(); }
    buffers_.clear();
  }
}

void* ScopedAllocator::allocate(size_t bytes)
{
  if (bytes == 0) return nullptr;

  ByteBuffer buffer = create_buffer<int8_t>(bytes, target_kind_, alignment_);

  void* ptr = buffer.ptr(0);

  buffers_[ptr] = buffer;
  return ptr;
}

void ScopedAllocator::deallocate(void* ptr)
{
  ByteBuffer buffer;
  auto finder = buffers_.find(ptr);
  assert(finder != buffers_.end());
  if (finder == buffers_.end()) {
    legate::log_legate.error("Invalid address during deallocation");
    LEGATE_ABORT;
  }

  buffer = finder->second;
  buffers_.erase(finder);
  buffer.destroy();
}

}  // namespace legate