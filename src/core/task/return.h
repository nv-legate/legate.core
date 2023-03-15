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
#include "core/utilities/typedefs.h"

namespace legate {

struct ReturnValue {
 public:
  ReturnValue(Legion::UntypedDeferredValue value, size_t size);

 public:
  ReturnValue(const ReturnValue&)            = default;
  ReturnValue& operator=(const ReturnValue&) = default;

 public:
  static ReturnValue unpack(const void* ptr, size_t size, Memory::Kind memory_kind);

 public:
  void* ptr();
  const void* ptr() const;
  size_t size() const { return size_; }
  bool is_device_value() const { return is_device_value_; }

 public:
  // Calls the Legion postamble with an instance
  void finalize(Legion::Context legion_context) const;

 private:
  Legion::UntypedDeferredValue value_{};
  size_t size_{0};
  bool is_device_value_{false};
};

struct ReturnedException {
 public:
  ReturnedException() {}
  ReturnedException(int32_t index, const std::string& error_message);

 public:
  bool raised() const { return raised_; }

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

 public:
  ReturnValue pack() const;

 private:
  bool raised_{false};
  int32_t index_{-1};
  std::string error_message_{};
};

struct ReturnValues {
 public:
  ReturnValues();
  ReturnValues(std::vector<ReturnValue>&& return_values);

 public:
  ReturnValues(const ReturnValues&)            = default;
  ReturnValues& operator=(const ReturnValues&) = default;

 public:
  ReturnValues(ReturnValues&&)            = default;
  ReturnValues& operator=(ReturnValues&&) = default;

 public:
  ReturnValue operator[](int32_t idx) const;

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

 public:
  static ReturnValue extract(Legion::Future future, uint32_t to_extract);

 public:
  // Calls the Legion postamble with an instance that packs all return values
  void finalize(Legion::Context legion_context) const;

 private:
  size_t buffer_size_{0};
  std::vector<ReturnValue> return_values_{};
};

}  // namespace legate
