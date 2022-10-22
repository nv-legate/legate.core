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

#include "core/mapping/operation.h"
#include "core/utilities/deserializer.h"

namespace legate {
namespace mapping {

using LegionTask = Legion::Task;
using LegionCopy = Legion::Copy;

using namespace Legion;
using namespace Legion::Mapping;

RegionField::RegionField(const RegionRequirement* req, int32_t dim, uint32_t idx, FieldID fid)
  : req_(req), dim_(dim), idx_(idx), fid_(fid)
{
}

bool RegionField::can_colocate_with(const RegionField& other) const
{
  auto* my_req    = get_requirement();
  auto* other_req = other.get_requirement();
  return my_req->region.get_tree_id() == other_req->region.get_tree_id();
}

Domain RegionField::domain(MapperRuntime* runtime, const MapperContext context) const
{
  return runtime->get_index_space_domain(context, get_index_space());
}

IndexSpace RegionField::get_index_space() const { return req_->region.get_index_space(); }

FutureWrapper::FutureWrapper(uint32_t idx, const Domain& domain) : idx_(idx), domain_(domain) {}

Domain FutureWrapper::domain() const { return domain_; }

Store::Store(int32_t dim,
             LegateTypeCode code,
             FutureWrapper future,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(true),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(-1),
    future_(future),
    transform_(std::forward<decltype(transform)>(transform))
{
}

Store::Store(Legion::Mapping::MapperRuntime* runtime,
             const Legion::Mapping::MapperContext context,
             int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             const RegionField& region_field,
             bool is_output_store,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_output_store_(is_output_store),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(region_field),
    transform_(std::forward<decltype(transform)>(transform)),
    runtime_(runtime),
    context_(context)
{
}

bool Store::can_colocate_with(const Store& other) const
{
  if (is_future() || other.is_future())
    return false;
  else if (is_reduction() || other.is_reduction())
    return false;
  return region_field_.can_colocate_with(other.region_field_);
}

const RegionField& Store::region_field() const
{
  assert(!is_future());
  return region_field_;
}

const FutureWrapper& Store::future() const
{
  assert(is_future());
  return future_;
}

Domain Store::domain() const
{
  assert(!unbound());
  auto result = is_future_ ? future_.domain() : region_field_.domain(runtime_, context_);
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

Task::Task(const LegionTask* task,
           const LibraryContext& library,
           MapperRuntime* runtime,
           const MapperContext context)
  : task_(task), library_(library)
{
  TaskDeserializer dez(task, runtime, context);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();
}

int64_t Task::task_id() const { return library_.get_local_task_id(task_->task_id); }

Copy::Copy(const LegionCopy* copy, MapperRuntime* runtime, const MapperContext context)
  : copy_(copy)
{
  CopyDeserializer dez(copy->mapper_data,
                       copy->mapper_data_size,
                       {copy->src_requirements,
                        copy->dst_requirements,
                        copy->src_indirect_requirements,
                        copy->dst_indirect_requirements},
                       runtime,
                       context);
  inputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<std::vector<Store>>();
#ifdef DEBUG_LEGATE
  for (auto& input : inputs_) assert(!input.is_future());
  for (auto& output : outputs_) assert(!output.is_future());
  for (auto& input_indirection : input_indirections_) assert(!input_indirection.is_future());
  for (auto& output_indirection : output_indirections_) assert(!output_indirection.is_future());
#endif
}

}  // namespace mapping
}  // namespace legate