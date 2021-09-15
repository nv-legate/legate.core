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
#include "mapping/task.h"

using LegionTask = Legion::Task;

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {

TaskDeserializer::TaskDeserializer(const LegionTask* task,
                                   const std::vector<PhysicalRegion>& regions)
  : BaseDeserializer(task), regions_{regions.data(), regions.size()}, outputs_()
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
  : BaseDeserializer(task),
    regions_{task->regions.data(), task->regions.size()},
    output_regions_{task->output_regions.data(), task->output_regions.size()},
    runtime_(runtime),
    context_(context)
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

void MapperDeserializer::_unpack(RegionField& value, bool is_output_region)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = RegionField(dim, is_output_region ? output_regions_[idx] : regions_[idx], fid);
}

}  // namespace mapping

}  // namespace legate
