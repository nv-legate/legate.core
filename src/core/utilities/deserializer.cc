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

#include "core/utilities/deserializer.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/task.h"

using LegionTask = Legion::Task;

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {

TaskDeserializer::TaskDeserializer(const LegionTask* task,
                                   const std::vector<PhysicalRegion>& regions)
  : BaseDeserializer(task),
    futures_{task->futures.data(), task->futures.size()},
    regions_{regions.data(), regions.size()},
    outputs_()
{
  auto runtime = Runtime::get_runtime();
  auto ctx     = Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);

  first_task_ = !task->is_index_space || (task->index_point == task->index_domain.lo());
}

void TaskDeserializer::_unpack(Store& value)
{
  auto is_future = unpack<bool>();
  auto dim       = unpack<int32_t>();
  auto code      = unpack<LegateTypeCode>();

  auto transform = unpack_transform();

  if (is_future) {
    auto fut = unpack<FutureWrapper>();
    value    = Store(dim, code, fut, transform);
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

void TaskDeserializer::_unpack(FutureWrapper& value)
{
  auto read_only   = unpack<bool>();
  auto has_storage = unpack<bool>();
  auto field_size  = unpack<int32_t>();

  auto point = unpack<std::vector<int64_t>>();
  Legion::Domain domain;
  domain.dim = static_cast<int32_t>(point.size());
  for (int32_t idx = 0; idx < domain.dim; ++idx) {
    domain.rect_data[idx]              = 0;
    domain.rect_data[idx + domain.dim] = point[idx] - 1;
  }

  Legion::Future future;
  if (has_storage) {
    future   = futures_[0];
    futures_ = futures_.subspan(1);
  }

  value = FutureWrapper(read_only, field_size, domain, future, has_storage && first_task_);
}

void TaskDeserializer::_unpack(RegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = RegionField(dim, regions_[idx], fid);
}

void TaskDeserializer::_unpack(OutputRegionField& value)
{
  auto dim = unpack<int32_t>();
  assert(dim == 1);
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = OutputRegionField(outputs_[idx], fid);
}

namespace mapping {

MapperDeserializer::MapperDeserializer(const LegionTask* task,
                                       MapperRuntime* runtime,
                                       MapperContext context)
  : BaseDeserializer(task), runtime_(runtime), context_(context)
{
  first_task_ = false;
}

void MapperDeserializer::_unpack(Store& value)
{
  auto is_future = unpack<bool>();
  auto dim       = unpack<int32_t>();
  auto code      = unpack<LegateTypeCode>();

  auto transform = unpack_transform();

  if (is_future) {
    auto fut = unpack<FutureWrapper>();
    value    = Store(dim, code, fut, std::move(transform));
  } else {
    auto is_output_region = dim < 0;
    auto redop_id         = unpack<int32_t>();
    RegionField rf;
    _unpack(rf, is_output_region);
    value =
      Store(runtime_, context_, dim, code, redop_id, rf, is_output_region, std::move(transform));
  }
}

void MapperDeserializer::_unpack(FutureWrapper& value)
{
  // We still need to deserialize these fields to get to the domain
  unpack<bool>();
  unpack<bool>();
  unpack<int32_t>();

  auto point = unpack<std::vector<int64_t>>();
  Legion::Domain domain;
  domain.dim = static_cast<int32_t>(point.size());
  for (int32_t idx = 0; idx < domain.dim; ++idx) {
    domain.rect_data[idx]              = 0;
    domain.rect_data[idx + domain.dim] = point[idx] - 1;
  }

  value = FutureWrapper(domain);
}

void MapperDeserializer::_unpack(RegionField& value, bool is_output_region)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = RegionField(task_, dim, idx, fid);
}

}  // namespace mapping

}  // namespace legate
