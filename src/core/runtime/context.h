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

#include "legion.h"

#include "core/task/return.h"

namespace legate {

class Store;
class Scalar;

struct ResourceConfig {
  int64_t max_tasks{1000000};
  int64_t max_mappers{1};
  int64_t max_reduction_ops{0};
  int64_t max_projections{0};
  int64_t max_shardings{0};
};

class ResourceScope {
 public:
  ResourceScope() = default;
  ResourceScope(int64_t base, int64_t max) : base_(base), max_(max) {}

 public:
  ResourceScope(const ResourceScope&) = default;

 public:
  int64_t translate(int64_t local_resource_id) const { return base_ + local_resource_id; }
  int64_t invert(int64_t resource_id) const
  {
    assert(in_scope(resource_id));
    return resource_id - base_;
  }

 public:
  bool valid() const { return base_ != -1; }
  bool in_scope(int64_t resource_id) const
  {
    return base_ <= resource_id && resource_id < base_ + max_;
  }

 private:
  int64_t base_{-1};
  int64_t max_{-1};
};

class LibraryContext {
 public:
  LibraryContext(Legion::Runtime* runtime,
                 const std::string& library_name,
                 const ResourceConfig& config);

 public:
  LibraryContext(const LibraryContext&) = default;

 public:
  const std::string& get_library_name() const;

 public:
  Legion::TaskID get_task_id(int64_t local_task_id) const;
  Legion::MapperID get_mapper_id(int64_t local_mapper_id) const;
  Legion::ReductionOpID get_reduction_op_id(int64_t local_redop_id) const;
  Legion::ProjectionID get_projection_id(int64_t local_proj_id) const;
  Legion::ShardingID get_sharding_id(int64_t local_shard_id) const;

 public:
  int64_t get_local_task_id(Legion::TaskID task_id) const;
  int64_t get_local_mapper_id(Legion::MapperID mapper_id) const;
  int64_t get_local_reduction_op_id(Legion::ReductionOpID redop_id) const;
  int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

 public:
  bool valid_task_id(Legion::TaskID task_id) const;
  bool valid_mapper_id(Legion::MapperID mapper_id) const;
  bool valid_reduction_op_id(Legion::ReductionOpID redop_id) const;
  bool valid_projection_id(Legion::ProjectionID proj_id) const;
  bool valid_sharding_id(Legion::ShardingID shard_id) const;

 private:
  const std::string library_name_;
  ResourceScope task_scope_;
  ResourceScope mapper_scope_;
  ResourceScope redop_scope_;
  ResourceScope proj_scope_;
  ResourceScope shard_scope_;
};

// A thin context layer on top of the Legion runtime, primarily designed to hide verbosity
// of the Legion API.
class TaskContext {
 public:
  TaskContext(const Legion::Task* task,
              const std::vector<Legion::PhysicalRegion>& regions,
              Legion::Context context,
              Legion::Runtime* runtime);

 public:
  std::vector<Store>& inputs() { return inputs_; }
  std::vector<Store>& outputs() { return outputs_; }
  std::vector<Store>& reductions() { return reductions_; }
  std::vector<Scalar>& scalars() { return scalars_; }

 public:
  ReturnValues pack_return_values() const;

 private:
  const Legion::Task* task_;
  const std::vector<Legion::PhysicalRegion>& regions_;
  Legion::Context context_;
  Legion::Runtime* runtime_;

 private:
  std::vector<Store> inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
};

}  // namespace legate
