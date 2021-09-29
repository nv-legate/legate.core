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

#include "utilities/deserializer.h"
#include "data/scalar.h"
#include "data/store.h"

namespace legate {

using namespace Legion;

Deserializer::Deserializer(const Task* task, const std::vector<PhysicalRegion>& regions)
  : regions_{regions.data(), regions.size()},
    futures_{task->futures.data(), task->futures.size()},
    task_args_{static_cast<const int8_t*>(task->args), task->arglen},
    outputs_()
{
  auto runtime = Runtime::get_runtime();
  auto ctx     = Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

void Deserializer::_unpack(LegateTypeCode& value)
{
  value = static_cast<LegateTypeCode>(unpack<int32_t>());
}


void Deserializer::_unpack(FusionMetadata& metadata){
    metadata.isFused = unpack<bool>();
    if (!metadata.isFused){
        return;
    }
    //exit out if the this is not a fused op
    metadata.nOps = unpack<int32_t>();
    metadata.nBuffers = unpack<int32_t>();
    int nOps = metadata.nOps;
    int nBuffers = metadata.nBuffers; 

    metadata.inputStarts.resize(nOps+1);
    metadata.outputStarts.resize(nOps+1);
    metadata.offsetStarts.resize(nOps+1);
    metadata.offsets.resize(nBuffers+1);
    metadata.reductionStarts.resize(nOps+1);
    metadata.scalarStarts.resize(nOps+1);
    metadata.opIDs.resize(nOps);
    //TODO: wrap this up to reuse code`
    for (int i=0; i<nOps+1; i++)
    {
        metadata.inputStarts[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nOps+1; i++)
    {
        metadata.outputStarts[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nOps+1; i++)
    {
        metadata.offsetStarts[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nBuffers; i++)
    {
        metadata.offsets[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nOps+1; i++)
    {
        metadata.reductionStarts[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nOps+1; i++)
    {
        metadata.scalarStarts[i] = unpack<int32_t>();
    }   
    for (int i=0; i<nOps; i++)
    {
        metadata.opIDs[i] = unpack<int32_t>();
    }   
}

void Deserializer::_unpack(Store& value)
{
  auto is_future = unpack<bool>();
  auto dim       = unpack<int32_t>();
  auto code      = unpack<LegateTypeCode>();

  auto transform = unpack_transform();

  if (is_future) {
    auto fut = unpack<FutureWrapper>();
    value    = Store(dim, code, fut, std::move(transform));
  } else if (dim >= 0) {
    auto redop_id = unpack<int32_t>();
    auto rf       = unpack<RegionField>();
    value         = Store(dim, code, redop_id, std::move(rf), std::move(transform));
  } else {
    auto redop_id = unpack<int32_t>();
    assert(redop_id == -1);
    auto out = unpack<OutputRegionField>();
    value    = Store(code, std::move(out), std::move(transform));
  }
}

void Deserializer::_unpack(Scalar& value)
{
  auto tuple = unpack<bool>();
  auto code  = unpack<LegateTypeCode>();
  value      = Scalar(tuple, code, task_args_.ptr());
  task_args_ = task_args_.subspan(value.size());
}

void Deserializer::_unpack(FutureWrapper& value)
{
  auto future = futures_[0];
  futures_    = futures_.subspan(1);

  auto point = unpack<std::vector<int64_t>>();
  Domain domain;
  domain.dim = static_cast<int32_t>(point.size());
  for (int32_t idx = 0; idx < domain.dim; ++idx) {
    domain.rect_data[idx]              = 0;
    domain.rect_data[idx + domain.dim] = point[idx] - 1;
  }

  value = FutureWrapper(domain, future);
}

void Deserializer::_unpack(RegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = RegionField(dim, regions_[idx], fid, idx);
}

void Deserializer::_unpack(OutputRegionField& value)
{
  auto dim = unpack<int32_t>();
  assert(dim == 1);
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = OutputRegionField(outputs_[idx], fid, idx);
}

std::unique_ptr<StoreTransform> Deserializer::unpack_transform()
{
  int32_t code = unpack<int32_t>();
  switch (code) {
    case -1: {
      return nullptr;
    }
    case LEGATE_CORE_TRANSFORM_SHIFT: {
      auto dim    = unpack<int32_t>();
      auto offset = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_unique<Shift>(dim, offset, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROMOTE: {
      auto extra_dim = unpack<int32_t>();
      auto dim_size  = unpack<int64_t>();
      auto parent    = unpack_transform();
      return std::make_unique<Promote>(extra_dim, dim_size, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROJECT: {
      auto dim    = unpack<int32_t>();
      auto coord  = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_unique<Project>(dim, coord, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
      auto axes   = unpack<std::vector<int32_t>>();
      auto parent = unpack_transform();
      return std::make_unique<Transpose>(std::move(axes), std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
      auto dim    = unpack<int32_t>();
      auto sizes  = unpack<std::vector<int64_t>>();
      auto parent = unpack_transform();
      return std::make_unique<Delinearize>(dim, std::move(sizes), std::move(parent));
    }
  }
  assert(false);
  return nullptr;
}

}  // namespace legate
