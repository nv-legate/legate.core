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

#include "mapping/task.h"
#include "utilities/deserializer.h"

namespace legate {
namespace mapping {

using LegionTask = Legion::Task;

using namespace Legion;
using namespace Legion::Mapping;

RegionField::RegionField(int32_t dim, const RegionRequirement& req, FieldID fid)
  : dim_(dim), req_(req), fid_(fid)
{
}

Domain RegionField::domain(MapperRuntime* runtime, const MapperContext context) const
{
  return runtime->get_index_space_domain(context, req_.region.get_index_space());
}

Store::Store(int32_t dim,
             LegateTypeCode code,
             FutureWrapper future,
             std::unique_ptr<StoreTransform> transform)
  : is_future_(true),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(-1),
    future_(future),
    transform_(std::move(transform))
{
}

Store::Store(Legion::Mapping::MapperRuntime* runtime,
             const Legion::Mapping::MapperContext context,
             int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             const RegionField& region_field,
             bool is_output_store,
             std::unique_ptr<StoreTransform> transform)
  : is_future_(false),
    is_output_store_(is_output_store),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(region_field),
    transform_(std::move(transform)),
    runtime_(runtime),
    context_(context)
{
}

Store::Store(Store&& store) noexcept
  : is_future_(store.is_future_),
    is_output_store_(store.is_output_store_),
    dim_(store.dim_),
    code_(store.code_),
    redop_id_(store.redop_id_),
    future_(store.future_),
    region_field_(store.region_field_),
    transform_(std::move(store.transform_)),
    runtime_(store.runtime_),
    context_(store.context_)
{
}

Store& Store::operator=(Store&& store) noexcept
{
  is_future_       = store.is_future_;
  is_output_store_ = store.is_output_store_;
  dim_             = store.dim_;
  code_            = store.code_;
  redop_id_        = store.redop_id_;
  future_          = store.future_;
  region_field_    = store.region_field_;
  transform_       = std::move(store.transform_);
  runtime_         = store.runtime_;
  context_         = store.context_;
}

Domain Store::domain() const
{
  assert(!unbound());
  auto result = is_future_ ? future_.domain() : region_field_.domain(runtime_, context_);
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

Task::Task(const LegionTask& task, MapperRuntime* runtime, const MapperContext context)
  : task_(task)
{
  MapperDeserializer dez(&task, runtime, context);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();
}

}  // namespace mapping
}  // namespace legate
