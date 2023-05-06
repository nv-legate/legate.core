/* Copyright 2022 NVIDIA Corporation
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

#include "core/mapping/store.h"

namespace legate {
namespace mapping {

RegionField::RegionField(const Legion::RegionRequirement* req,
                         int32_t dim,
                         uint32_t idx,
                         Legion::FieldID fid)
  : req_(req), dim_(dim), idx_(idx), fid_(fid)
{
}

bool RegionField::can_colocate_with(const RegionField& other) const
{
  auto* my_req    = get_requirement();
  auto* other_req = other.get_requirement();
  return my_req->region.get_tree_id() == other_req->region.get_tree_id();
}

Domain RegionField::domain(Legion::Mapping::MapperRuntime* runtime,
                           const Legion::Mapping::MapperContext context) const
{
  return runtime->get_index_space_domain(context, get_index_space());
}

Legion::IndexSpace RegionField::get_index_space() const { return req_->region.get_index_space(); }

FutureWrapper::FutureWrapper(uint32_t idx, const Domain& domain) : idx_(idx), domain_(domain) {}

Domain FutureWrapper::domain() const { return domain_; }

Store::Store(int32_t dim,
             std::unique_ptr<Type> type,
             FutureWrapper future,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(true),
    is_unbound_store_(false),
    dim_(dim),
    type_(std::move(type)),
    redop_id_(-1),
    future_(future),
    transform_(std::forward<decltype(transform)>(transform))
{
}

Store::Store(Legion::Mapping::MapperRuntime* runtime,
             const Legion::Mapping::MapperContext context,
             int32_t dim,
             std::unique_ptr<Type> type,
             int32_t redop_id,
             const RegionField& region_field,
             bool is_unbound_store,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_unbound_store_(is_unbound_store),
    dim_(dim),
    type_(std::move(type)),
    redop_id_(redop_id),
    region_field_(region_field),
    transform_(std::forward<decltype(transform)>(transform)),
    runtime_(runtime),
    context_(context)
{
}

Store::Store(Legion::Mapping::MapperRuntime* runtime,
             const Legion::Mapping::MapperContext context,
             const Legion::RegionRequirement* requirement)
  : is_future_(false),
    is_unbound_store_(false),
    dim_(requirement->region.get_dim()),
    type_(nullptr),
    redop_id_(-1),
    runtime_(runtime),
    context_(context)
{
  region_field_ = RegionField(requirement, dim_, 0, requirement->instance_fields.front());
}

bool Store::can_colocate_with(const Store& other) const
{
  if (is_future() || other.is_future())
    return false;
  else if (unbound() || other.unbound())
    return false;
  else if (is_reduction() || other.is_reduction())
    return redop() == other.redop() && region_field_.can_colocate_with(other.region_field_);
  return region_field_.can_colocate_with(other.region_field_);
}

const RegionField& Store::region_field() const
{
#ifdef DEBUG_LEGATE
  assert(!is_future());
#endif
  return region_field_;
}

const FutureWrapper& Store::future() const
{
#ifdef DEBUG_LEGATE
  assert(is_future());
#endif
  return future_;
}

RegionField::Id Store::unique_region_field_id() const { return region_field().unique_id(); }

uint32_t Store::requirement_index() const { return region_field().index(); }

uint32_t Store::future_index() const { return future().index(); }

Domain Store::domain() const
{
  assert(!unbound());
  auto result = is_future_ ? future_.domain() : region_field_.domain(runtime_, context_);
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

}  // namespace mapping
}  // namespace legate
