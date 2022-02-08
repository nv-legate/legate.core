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

#include <memory>

#include "legion.h"

#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/task.h"
#include "core/utilities/span.h"
#include "core/utilities/type_traits.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"

namespace legate {

template <typename Deserializer>
class BaseDeserializer {
 public:
  BaseDeserializer(const Legion::Task* task);

 public:
  template <typename T>
  T unpack()
  {
    T value;
    static_cast<Deserializer*>(this)->_unpack(value);
    return std::move(value);
  }

 public:
  template <typename T, std::enable_if_t<legate_type_code_of<T> != MAX_TYPE_NUMBER>* = nullptr>
  void _unpack(T& value)
  {
    value      = *reinterpret_cast<const T*>(task_args_.ptr());
    task_args_ = task_args_.subspan(sizeof(T));
  }

 public:
  template <typename T>
  void _unpack(std::vector<T>& values)
  {
    auto size = unpack<uint32_t>();
    for (uint32_t idx = 0; idx < size; ++idx) values.push_back(unpack<T>());
  }

 public:
  void _unpack(LegateTypeCode& value);
  void _unpack(Scalar& value);

 protected:
  std::shared_ptr<StoreTransform> unpack_transform();

 protected:
  const Legion::Task* task_;
  bool first_task_;

 private:
  Span<const int8_t> task_args_;
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

 public:
  using BaseDeserializer::_unpack;

 public:
  void _unpack(Store& value);
  void _unpack(FutureWrapper& value);
  void _unpack(RegionField& value);
  void _unpack(OutputRegionField& value);

 private:
  Span<const Legion::Future> futures_;
  Span<const Legion::PhysicalRegion> regions_;
  std::vector<Legion::OutputRegion> outputs_;
};

namespace mapping {

class MapperDeserializer : public BaseDeserializer<MapperDeserializer> {
 public:
  MapperDeserializer(const Legion::Task* task,
                     Legion::Mapping::MapperRuntime* runtime,
                     Legion::Mapping::MapperContext context);

 public:
  using BaseDeserializer::_unpack;

 public:
  void _unpack(Store& value);
  void _unpack(FutureWrapper& value);
  void _unpack(RegionField& value, bool is_output_region);

 private:
  Legion::Mapping::MapperRuntime* runtime_;
  Legion::Mapping::MapperContext context_;
};

}  // namespace mapping

}  // namespace legate

#include "core/utilities/deserializer.inl"
