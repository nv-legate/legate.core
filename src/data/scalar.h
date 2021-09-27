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

#pragma once

#include "utilities/span.h"
#include "utilities/typedefs.h"
#include "utilities/makeshift_serializer.h"

namespace legate {

class Scalar {
 public:
  Scalar()              = default;
  Scalar(const Scalar&) = default;
  Scalar(bool tuple, LegateTypeCode code, const void* data);

 public:
  Scalar& operator=(const Scalar&) = default;

 public:
  bool is_tuple() const { return tuple_; }
  size_t size() const;

 public:
  template <typename VAL>
  VAL value() const;
  template <typename VAL>
  Span<const VAL> values() const;

 private:
  bool tuple_{false};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  const void* data_;

  friend class MakeshiftSerializer;
};

}  // namespace legate

#include "scalar.inl"
