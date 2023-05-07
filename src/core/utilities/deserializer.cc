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
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"

#include "legion/legion_c.h"
#include "legion/legion_c_util.h"

namespace legate {

TaskDeserializer::TaskDeserializer(const Legion::Task* task,
                                   const std::vector<Legion::PhysicalRegion>& regions)
  : BaseDeserializer(task->args, task->arglen),
    futures_{task->futures.data(), task->futures.size()},
    regions_{regions.data(), regions.size()},
    outputs_()
{
  auto runtime = Legion::Runtime::get_runtime();
  auto ctx     = Legion::Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);

  first_task_ = !task->is_index_space || (task->index_point == task->index_domain.lo());
}

void TaskDeserializer::_unpack(Store& value)
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();

  auto transform = unpack_transform();

  if (is_future) {
    auto redop_id = unpack<int32_t>();
    auto fut      = unpack<FutureWrapper>();
    if (redop_id != -1 && !first_task_) fut.initialize_with_identity(redop_id);
    value = Store(dim, std::move(type), redop_id, fut, std::move(transform));
  } else if (!is_output_region) {
    auto redop_id = unpack<int32_t>();
    auto rf       = unpack<RegionField>();
    value         = Store(dim, std::move(type), redop_id, std::move(rf), std::move(transform));
  } else {
    auto redop_id = unpack<int32_t>();
    assert(redop_id == -1);
    auto out = unpack<UnboundRegionField>();
    value    = Store(dim, std::move(type), std::move(out), std::move(transform));
  }
}

void TaskDeserializer::_unpack(FutureWrapper& value)
{
  auto read_only   = unpack<bool>();
  auto has_storage = unpack<bool>();
  auto field_size  = unpack<uint32_t>();

  auto point = unpack<std::vector<int64_t>>();
  Domain domain;
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

void TaskDeserializer::_unpack(UnboundRegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = UnboundRegionField(outputs_[idx], fid);
}

void TaskDeserializer::_unpack(comm::Communicator& value)
{
  auto future = futures_[0];
  futures_    = futures_.subspan(1);
  value       = comm::Communicator(future);
}

void TaskDeserializer::_unpack(Legion::PhaseBarrier& barrier)
{
  auto future   = futures_[0];
  futures_      = futures_.subspan(1);
  auto barrier_ = future.get_result<legion_phase_barrier_t>();
  barrier       = Legion::CObjectWrapper::unwrap(barrier_);
}

namespace mapping {

MapperDataDeserializer::MapperDataDeserializer(const Legion::Mappable* mappable)
  : BaseDeserializer(mappable->mapper_data, mappable->mapper_data_size)
{
}

TaskDeserializer::TaskDeserializer(const Legion::Task* task,
                                   Legion::Mapping::MapperRuntime* runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer(task->args, task->arglen),
    task_(task),
    runtime_(runtime),
    context_(context),
    future_index_(0)
{
  first_task_ = false;
}

void TaskDeserializer::_unpack(Store& value)
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();

  auto transform = unpack_transform();

  if (is_future) {
    // We still need to parse the reduction op
    unpack<int32_t>();
    auto fut = unpack<FutureWrapper>();
    value    = Store(dim, std::move(type), fut, std::move(transform));
  } else {
    auto redop_id = unpack<int32_t>();
    RegionField rf;
    _unpack(rf, is_output_region);
    value = Store(runtime_,
                  context_,
                  dim,
                  std::move(type),
                  redop_id,
                  rf,
                  is_output_region,
                  std::move(transform));
  }
}

void TaskDeserializer::_unpack(FutureWrapper& value)
{
  // We still need to deserialize these fields to get to the domain
  unpack<bool>();
  unpack<bool>();
  unpack<uint32_t>();

  auto point = unpack<std::vector<int64_t>>();
  Domain domain;
  domain.dim = static_cast<int32_t>(point.size());
  for (int32_t idx = 0; idx < domain.dim; ++idx) {
    domain.rect_data[idx]              = 0;
    domain.rect_data[idx + domain.dim] = point[idx] - 1;
  }

  value = FutureWrapper(future_index_++, domain);
}

void TaskDeserializer::_unpack(RegionField& value, bool is_output_region)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  auto req = is_output_region ? &task_->output_regions[idx] : &task_->regions[idx];
  value    = RegionField(req, dim, idx, fid);
}

CopyDeserializer::CopyDeserializer(const Legion::Copy* copy,
                                   std::vector<ReqsRef>&& all_requirements,
                                   Legion::Mapping::MapperRuntime* runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer(copy->mapper_data, copy->mapper_data_size),
    all_reqs_(std::forward<std::vector<ReqsRef>>(all_requirements)),
    curr_reqs_(all_reqs_.begin()),
    runtime_(runtime),
    context_(context),
    req_index_offset_(0)
{
}

void CopyDeserializer::next_requirement_list()
{
#ifdef DEBUG_LEGATE
  assert(curr_reqs_ != all_reqs_.end());
#endif
  req_index_offset_ += curr_reqs_->get().size();
  ++curr_reqs_;
}

void CopyDeserializer::_unpack(Store& value)
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();

  auto transform = unpack_transform();

#ifdef DEBUG_LEGATE
  assert(!is_future && !is_output_region);
#endif
  auto redop_id = unpack<int32_t>();
  RegionField rf;
  _unpack(rf);
  value = Store(
    runtime_, context_, dim, std::move(type), redop_id, rf, is_output_region, std::move(transform));
}

void CopyDeserializer::_unpack(RegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  auto req = &curr_reqs_->get()[idx];
  value    = RegionField(req, dim, idx + req_index_offset_, fid);
}

}  // namespace mapping

}  // namespace legate
