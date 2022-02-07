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

#include <vector>

namespace legate {

using ReturnValue = std::pair<const void*, size_t>;

struct ReturnValues {
 public:
  ReturnValues();
  ReturnValues(std::vector<ReturnValue>&& return_values);

 public:
  ReturnValues(const ReturnValues&) = default;
  ReturnValues& operator=(const ReturnValues&) = default;

 public:
  ReturnValues(ReturnValues&&) = default;
  ReturnValues& operator=(ReturnValues&&) = default;

 public:
  ReturnValue operator[](int32_t idx) const;

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

 private:
  size_t buffer_size_{0};
  std::vector<ReturnValue> return_values_{};
};

}  // namespace legate
