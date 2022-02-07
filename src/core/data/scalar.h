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

#include "core/utilities/span.h"
#include "core/utilities/type_traits.h"
#include "core/utilities/typedefs.h"

namespace legate {

class Scalar {
 public:
  Scalar() = default;
  Scalar(const Scalar& other);
  Scalar(bool tuple, LegateTypeCode code, const void* data);
  ~Scalar();

 public:
  template <typename T>
  Scalar(T value);
  template <typename T>
  Scalar(const std::vector<T>& values);

 public:
  Scalar& operator=(const Scalar& other);

 private:
  void copy(const Scalar& other);

 public:
  bool is_tuple() const { return tuple_; }
  size_t size() const;

 public:
  template <typename VAL>
  VAL value() const;
  template <typename VAL>
  Span<const VAL> values() const;
  const void* ptr() const { return data_; }

 private:
  bool own_{false};
  bool tuple_{false};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  const void* data_;
};

}  // namespace legate

#include "core/data/scalar.inl"
