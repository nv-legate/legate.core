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

#include "legion.h"

#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/runtime/context.h"
#include "core/utilities/deserializer.h"

namespace legate {

LibraryContext::LibraryContext(Legion::Runtime* runtime,
                               const std::string& library_name,
                               const ResourceConfig& config)
  : library_name_(library_name)
{
  task_scope_ = ResourceScope(
    runtime->generate_library_task_ids(library_name.c_str(), config.max_tasks), config.max_tasks);
  mapper_scope_ =
    ResourceScope(runtime->generate_library_mapper_ids(library_name.c_str(), config.max_mappers),
                  config.max_mappers);
  redop_scope_ = ResourceScope(
    runtime->generate_library_reduction_ids(library_name.c_str(), config.max_reduction_ops),
    config.max_reduction_ops);
  proj_scope_ = ResourceScope(
    runtime->generate_library_projection_ids(library_name.c_str(), config.max_projections),
    config.max_projections);
  shard_scope_ = ResourceScope(
    runtime->generate_library_sharding_ids(library_name.c_str(), config.max_shardings),
    config.max_shardings);
}

const std::string& LibraryContext::get_library_name() const { return library_name_; }

Legion::TaskID LibraryContext::get_task_id(int64_t local_task_id) const
{
  assert(task_scope_.valid());
  return task_scope_.translate(local_task_id);
}

Legion::MapperID LibraryContext::get_mapper_id(int64_t local_mapper_id) const
{
  assert(mapper_scope_.valid());
  return mapper_scope_.translate(local_mapper_id);
}

Legion::ReductionOpID LibraryContext::get_reduction_op_id(int64_t local_redop_id) const
{
  assert(redop_scope_.valid());
  return redop_scope_.translate(local_redop_id);
}

Legion::ProjectionID LibraryContext::get_projection_id(int64_t local_proj_id) const
{
  if (local_proj_id == 0)
    return 0;
  else {
    assert(proj_scope_.valid());
    return proj_scope_.translate(local_proj_id);
  }
}

Legion::ShardingID LibraryContext::get_sharding_id(int64_t local_shard_id) const
{
  assert(shard_scope_.valid());
  return shard_scope_.translate(local_shard_id);
}

int64_t LibraryContext::get_local_task_id(Legion::TaskID task_id) const
{
  assert(task_scope_.valid());
  return task_scope_.invert(task_id);
}

int64_t LibraryContext::get_local_mapper_id(Legion::MapperID mapper_id) const
{
  assert(mapper_scope_.valid());
  return mapper_scope_.invert(mapper_id);
}

int64_t LibraryContext::get_local_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  assert(redop_scope_.valid());
  return redop_scope_.invert(redop_id);
}

int64_t LibraryContext::get_local_projection_id(Legion::ProjectionID proj_id) const
{
  if (proj_id == 0)
    return 0;
  else {
    assert(proj_scope_.valid());
    return proj_scope_.invert(proj_id);
  }
}

int64_t LibraryContext::get_local_sharding_id(Legion::ShardingID shard_id) const
{
  assert(shard_scope_.valid());
  return shard_scope_.invert(shard_id);
}

bool LibraryContext::valid_task_id(Legion::TaskID task_id) const
{
  return task_scope_.in_scope(task_id);
}

bool LibraryContext::valid_mapper_id(Legion::MapperID mapper_id) const
{
  return mapper_scope_.in_scope(mapper_id);
}

bool LibraryContext::valid_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  return redop_scope_.in_scope(redop_id);
}

bool LibraryContext::valid_projection_id(Legion::ProjectionID proj_id) const
{
  return proj_scope_.in_scope(proj_id);
}

bool LibraryContext::valid_sharding_id(Legion::ShardingID shard_id) const
{
  return shard_scope_.in_scope(shard_id);
}

TaskContext::TaskContext(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  : task_(task), regions_(regions), context_(context), runtime_(runtime)
{
  TaskDeserializer dez(task, regions);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();
}

ReturnValues TaskContext::pack_return_values() const
{
  std::vector<ReturnValue> return_values;

  for (auto& output : outputs_) {
    if (!output.is_future()) continue;
    return_values.push_back(output.pack());
  }
  for (auto& reduction : reductions_) {
    if (!reduction.is_future()) continue;
    return_values.push_back(reduction.pack());
  }

  return ReturnValues(std::move(return_values));
}

}  // namespace legate
