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
// Must be included after legion.h
#include "legate_defines.h"

#include "core/comm/communicator.h"
#include "core/task/return.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::Context
 */

namespace legate {

namespace mapping {

class LegateMapper;

}  // namespace mapping

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
  LibraryContext(const std::string& library_name, const ResourceConfig& config);

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

 public:
  template <typename REDOP>
  void register_reduction_operator();
  void register_mapper(std::unique_ptr<mapping::LegateMapper> mapper,
                       int64_t local_mapper_id = 0) const;

 private:
  Legion::Runtime* runtime_;
  const std::string library_name_;
  ResourceScope task_scope_;
  ResourceScope mapper_scope_;
  ResourceScope redop_scope_;
  ResourceScope proj_scope_;
  ResourceScope shard_scope_;
};

/**
 * @brief A task context that contains task arguments and communicators
 */
class TaskContext {
 public:
  TaskContext(const Legion::Task* task,
              const std::vector<Legion::PhysicalRegion>& regions,
              Legion::Context context,
              Legion::Runtime* runtime);

 public:
  /**
   * @brief Returns input stores of the task
   *
   * @return Vector of input stores
   */
  std::vector<Store>& inputs() { return inputs_; }
  /**
   * @brief Returns output stores of the task
   *
   * @return Vector of output stores
   */
  std::vector<Store>& outputs() { return outputs_; }
  /**
   * @brief Returns reduction stores of the task
   *
   * @return Vector of reduction stores
   */
  std::vector<Store>& reductions() { return reductions_; }
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalar objects
   */
  std::vector<Scalar>& scalars() { return scalars_; }
  /**
   * @brief Returns communicators of the task
   *
   * @return Vector of communicator objects
   */
  std::vector<comm::Communicator>& communicators() { return comms_; }

 public:
  /**
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  bool is_single_task() const;
  /**
   * @brief Indicates whether the task is allowed to raise an exception
   *
   * @return true The task can raise an exception
   * @return false The task must not raise an exception
   */
  bool can_raise_exception() const { return can_raise_exception_; }
  /**
   * @brief Returns the point of the task. A 0D point will be returned for a single task.
   *
   * @return The point of the task
   */
  DomainPoint get_task_index() const;
  /**
   * @brief Returns the task group's launch domain. A single task returns an empty domain
   *
   * @return The task group's launch domain
   */
  Domain get_launch_domain() const;

 public:
  /**
   * @brief Makes all of unbound output stores of this task empty
   */
  void make_all_unbound_stores_empty();
  ReturnValues pack_return_values() const;
  ReturnValues pack_return_values_with_exception(int32_t index,
                                                 const std::string& error_message) const;

 private:
  std::vector<ReturnValue> get_return_values() const;

 private:
  const Legion::Task* task_;
  const std::vector<Legion::PhysicalRegion>& regions_;
  Legion::Context context_;
  Legion::Runtime* runtime_;

 private:
  std::vector<Store> inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
  std::vector<comm::Communicator> comms_;
  bool can_raise_exception_;
};

}  // namespace legate

#include "core/runtime/context.inl"
