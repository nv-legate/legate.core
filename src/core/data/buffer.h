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

namespace legate {

template <typename VAL, int32_t DIM = 1>
using Buffer = Legion::DeferredBuffer<VAL, DIM>;

template <typename VAL, int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Legion::Point<DIM>& extents,
                               Legion::Memory::Kind kind = Legion::Memory::Kind::NO_MEMKIND,
                               size_t alignment          = 16)
{
  using namespace Legion;
  if (Memory::Kind::NO_MEMKIND == kind) {
    auto proc = Processor::get_executing_processor();
    kind      = proc.kind() == Processor::Kind::TOC_PROC ? Memory::Kind::GPU_FB_MEM
                                                         : Memory::Kind::SYSTEM_MEM;
  }
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
