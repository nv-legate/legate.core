/* Copyright 2021 NVIDIA Corporation
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

#include "task/return.h"
#ifdef LEGATE_USE_CUDA
#include <cuda.h>
#endif

using namespace Legion;

namespace legate {

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values)
  : return_values_(std::move(return_values))
{
  for (auto& ret : return_values_) buffer_size_ += ret.second;
}

size_t ReturnValues::legion_buffer_size() const { return buffer_size_; }

void ReturnValues::legion_serialize(void* buffer) const
{
  auto ptr = static_cast<int8_t*>(buffer);
#ifdef LEGATE_USE_CUDA
  auto kind = Processor::get_executing_processor().kind();
  if (kind == Processor::TOC_PROC) cudaDeviceSynchronize();
#endif
  for (auto& ret : return_values_) {
    memcpy(ptr, ret.first, ret.second);
    ptr += ret.second;
  }
}

void ReturnValues::legion_deserialize(const void* buffer) { assert(false); }

}  // namespace legate
