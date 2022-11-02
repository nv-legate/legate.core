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
#include <tuple>

#include "core/data/scalar.h"
#include "core/data/transform.h"
#include "core/mapping/machine.h"
#include "core/mapping/store.h"
#include "core/runtime/context.h"

namespace legate {
namespace mapping {

class Mappable {
 protected:
  Mappable();

 public:
  Mappable(const Legion::Mappable* mappable);

 public:
  const mapping::MachineDesc& machine_desc() const { return machine_desc_; }
  uint32_t sharding_id() const { return sharding_id_; }

 protected:
  mapping::MachineDesc machine_desc_;
  uint32_t sharding_id_;
};

class Task : public Mappable {
 public:
  Task(const Legion::Task* task,
       const LibraryContext& library,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  int64_t task_id() const;

 public:
  const std::vector<Store>& inputs() const { return inputs_; }
  const std::vector<Store>& outputs() const { return outputs_; }
  const std::vector<Store>& reductions() const { return reductions_; }
  const std::vector<Scalar>& scalars() const { return scalars_; }

 public:
  Legion::DomainPoint point() const { return task_->index_point; }

 public:
  TaskTarget target() const;

 private:
  const LibraryContext& library_;
  const Legion::Task* task_;

 private:
  std::vector<Store> inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
};

class Copy : public Mappable {
 public:
  Copy(const Legion::Copy* copy,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  const std::vector<Store>& inputs() const { return inputs_; }
  const std::vector<Store>& outputs() const { return outputs_; }
  const std::vector<Store>& input_indirections() const { return input_indirections_; }
  const std::vector<Store>& output_indirections() const { return output_indirections_; }

 public:
  Legion::DomainPoint point() const { return copy_->index_point; }

 private:
  const Legion::Copy* copy_;

 private:
  std::vector<Store> inputs_;
  std::vector<Store> outputs_;
  std::vector<Store> input_indirections_;
  std::vector<Store> output_indirections_;
};

}  // namespace mapping
}  // namespace legate

#include "core/mapping/operation.inl"
