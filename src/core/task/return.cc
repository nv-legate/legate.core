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

#include "core/task/return.h"
#include "core/utilities/machine.h"
#ifdef LEGATE_USE_CUDA
#include <cuda.h>
#endif

using namespace Legion;

namespace legate {

ReturnValues::ReturnValues() {}

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values)
  : return_values_(std::move(return_values))
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
#ifdef LEGATE_USE_CUDA
  auto kind = Processor::get_executing_processor().kind();
  if (kind == Processor::TOC_PROC) cudaDeviceSynchronize();
#endif
  if (return_values_.size() > 1) {
    *reinterpret_cast<size_t*>(ptr) = return_values_.size();
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

}  // namespace legate
