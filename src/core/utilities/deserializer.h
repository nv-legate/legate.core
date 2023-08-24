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

#include "core/comm/communicator.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/machine.h"
#include "core/mapping/operation.h"
#include "core/type/type_traits.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"

namespace legate {

template <typename Deserializer>
class BaseDeserializer {
 public:
  BaseDeserializer(const void* args, size_t arglen);

 public:
  template <typename T>
  T unpack()
  {
    T value;
    static_cast<Deserializer*>(this)->_unpack(value);
    return std::move(value);
  }

 public:
  template <typename T, std::enable_if_t<legate_type_code_of<T> != Type::Code::INVALID>* = nullptr>
  void _unpack(T& value)
  {
    value = *reinterpret_cast<const T*>(args_.ptr());
    args_ = args_.subspan(sizeof(T));
  }

 public:
  template <typename T>
  void _unpack(std::vector<T>& values)
  {
    auto size = unpack<uint32_t>();
    values.reserve(size);
    for (uint32_t idx = 0; idx < size; ++idx) values.emplace_back(unpack<T>());
  }
  template <typename T1, typename T2>
  void _unpack(std::pair<T1, T2>& values)
  {
    values.first  = unpack<T1>();
    values.second = unpack<T2>();
  }

 public:
  void _unpack(Scalar& value);
  void _unpack(mapping::TaskTarget& value);
  void _unpack(mapping::ProcessorRange& value);
  void _unpack(mapping::MachineDesc& value);

 public:
  Span<const int8_t> current_args() const { return args_; }

 protected:
  std::shared_ptr<TransformStack> unpack_transform();
  std::unique_ptr<Type> unpack_type();

 protected:
  Span<const int8_t> args_;
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
  void _unpack(UnboundRegionField& value);
  void _unpack(comm::Communicator& value);
  void _unpack(Legion::PhaseBarrier& barrier);

 private:
  Span<const Legion::Future> futures_;
  Span<const Legion::PhysicalRegion> regions_;
  std::vector<Legion::OutputRegion> outputs_;
};

namespace mapping {

class MapperDataDeserializer : public BaseDeserializer<MapperDataDeserializer> {
 public:
  MapperDataDeserializer(const Legion::Mappable* mappable);

 public:
  using BaseDeserializer::_unpack;
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task,
                   Legion::Mapping::MapperRuntime* runtime,
                   Legion::Mapping::MapperContext context);

 public:
  using BaseDeserializer::_unpack;

 public:
  void _unpack(Store& value);
  void _unpack(FutureWrapper& value);
  void _unpack(RegionField& value, bool is_output_region);

 private:
  const Legion::Task* task_;
  Legion::Mapping::MapperRuntime* runtime_;
  Legion::Mapping::MapperContext context_;
};

class CopyDeserializer : public BaseDeserializer<CopyDeserializer> {
 private:
  using Requirements = std::vector<Legion::RegionRequirement>;
  using ReqsRef      = std::reference_wrapper<const Requirements>;

 public:
  CopyDeserializer(const Legion::Copy* copy,
                   std::vector<ReqsRef>&& all_requirements,
                   Legion::Mapping::MapperRuntime* runtime,
                   Legion::Mapping::MapperContext context);

 public:
  using BaseDeserializer::_unpack;

 public:
  void next_requirement_list();

 public:
  void _unpack(Store& value);
  void _unpack(RegionField& value);

 private:
  std::vector<ReqsRef> all_reqs_;
  std::vector<ReqsRef>::iterator curr_reqs_;
  Legion::Mapping::MapperRuntime* runtime_;
  Legion::Mapping::MapperContext context_;
  uint32_t req_index_offset_;
};

}  // namespace mapping

}  // namespace legate

#include "core/utilities/deserializer.inl"
