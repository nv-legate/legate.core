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

#include <memory>

#include "legion.h"

#include "legate_defines.h"
#include "utilities/span.h"
#include "utilities/type_traits.h"
#include "utilities/typedefs.h"

namespace legate {

class Store;
class StoreTransform;
class Scalar;
class FutureWrapper;
class RegionField;
class OutputRegionField;
struct FusionMetadata;

class Deserializer {
 public:
  Deserializer(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

 public:
  template <typename T>
  T unpack()
  {
    T value;
    _unpack(value);
    return std::move(value);
  }

 //void unpackFusionMetadata(bool& isFused);

 private:
  template <typename T, std::enable_if_t<legate_type_code_of<T> != MAX_TYPE_NUMBER>* = nullptr>
  void _unpack(T& value)
  {
    value      = *reinterpret_cast<const T*>(task_args_.ptr());
    task_args_ = task_args_.subspan(sizeof(T));
  }

 private:
  template <typename T>
  void _unpack(std::vector<T>& values)
  {
    auto size = unpack<uint32_t>();
    for (uint32_t idx = 0; idx < size; ++idx) values.push_back(unpack<T>());
  }

 private:
  void _unpack(LegateTypeCode& value);
  void _unpack(Store& value);
  void _unpack(Scalar& value);
  void _unpack(FutureWrapper& value);
  void _unpack(RegionField& value);
  void _unpack(OutputRegionField& value);
  void _unpack(FusionMetadata& value);

 private:
  std::unique_ptr<StoreTransform> unpack_transform();

 private:
  Span<const Legion::PhysicalRegion> regions_;
  Span<const Legion::Future> futures_;
  Span<const int8_t> task_args_;
  std::vector<Legion::OutputRegion> outputs_;
};

}  // namespace legate
