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

#include "deserializer.h"
#include "dispatch.h"

namespace legate {

using namespace Legion;

Deserializer::Deserializer(const Task *task, const std::vector<PhysicalRegion> &regions)
  : task_{task},
    regions_{regions.data(), regions.size()},
    futures_{task->futures.data(), task->futures.size()},
    deserializer_{task->args, task->arglen},
    outputs_()
{
  auto runtime = Runtime::get_runtime();
  auto ctx     = Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

void deserialize(Deserializer &ctx, __half &value) { value = ctx.deserializer_.unpack_half(); }

void deserialize(Deserializer &ctx, float &value) { value = ctx.deserializer_.unpack_float(); }

void deserialize(Deserializer &ctx, double &value) { value = ctx.deserializer_.unpack_double(); }

void deserialize(Deserializer &ctx, std::uint64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_uint();
}

void deserialize(Deserializer &ctx, std::uint32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_uint();
}

void deserialize(Deserializer &ctx, std::uint16_t &value)
{
  value = ctx.deserializer_.unpack_16bit_uint();
}

void deserialize(Deserializer &ctx, std::uint8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_uint();
}

void deserialize(Deserializer &ctx, std::int64_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int32_t &value)
{
  value = ctx.deserializer_.unpack_32bit_int();
}

void deserialize(Deserializer &ctx, std::int16_t &value)
{
  value = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, std::int8_t &value)
{
  value = ctx.deserializer_.unpack_8bit_int();
}

void deserialize(Deserializer &ctx, std::string &value)
{
  value = ctx.deserializer_.unpack_string();
}

void deserialize(Deserializer &ctx, bool &value) { value = ctx.deserializer_.unpack_bool(); }

void deserialize(Deserializer &ctx, LegateTypeCode &code)
{
  code = static_cast<LegateTypeCode>(ctx.deserializer_.unpack_32bit_int());
}

struct deserialize_untyped_point_fn {
  template <int N>
  UntypedPoint operator()(LegateDeserializer &ctx)
  {
    return UntypedPoint(ctx.unpack_point<N>());
  }
};

void deserialize(Deserializer &ctx, UntypedPoint &value)
{
  auto dim = ctx.deserializer_.unpack_32bit_int();
  if (dim < 0) return;
  value = dim_dispatch(dim, deserialize_untyped_point_fn{}, ctx.deserializer_);
}

void deserialize(Deserializer &ctx, DomainPoint &value)
{
  auto dim  = ctx.deserializer_.unpack_32bit_int();
  value.dim = dim;
  for (auto idx = 0; idx < dim; ++idx) value[idx] = ctx.deserializer_.unpack_64bit_int();
}

void deserialize(Deserializer &ctx, Shape &value) {}

std::unique_ptr<Transform> deserialize_transform(Deserializer &ctx)
{
  int32_t code = ctx.deserializer_.unpack_32bit_int();
  switch (code) {
    case -1: {
      return nullptr;
    }
    case LEGATE_CORE_TRANSFORM_SHIFT: {
      int32_t dim    = ctx.deserializer_.unpack_32bit_int();
      int32_t offset = ctx.deserializer_.unpack_64bit_int();
      auto parent    = deserialize_transform(ctx);
      return std::make_unique<Shift>(dim, offset, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROMOTE: {
      int32_t extra_dim = ctx.deserializer_.unpack_32bit_int();
      int32_t dim_size  = ctx.deserializer_.unpack_64bit_int();
      auto parent       = deserialize_transform(ctx);
      return std::make_unique<Promote>(extra_dim, dim_size, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROJECT: {
      int32_t dim   = ctx.deserializer_.unpack_32bit_int();
      int32_t coord = ctx.deserializer_.unpack_64bit_int();
      auto parent   = deserialize_transform(ctx);
      return std::make_unique<Project>(dim, coord, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
      std::vector<int32_t> axes;
      deserialize(ctx, axes);
      auto parent = deserialize_transform(ctx);
      return std::make_unique<Transpose>(std::move(axes), std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
      int32_t dim = ctx.deserializer_.unpack_32bit_int();
      std::vector<int64_t> sizes;
      deserialize(ctx, sizes);
      auto parent = deserialize_transform(ctx);
      return std::make_unique<Delinearize>(dim, std::move(sizes), std::move(parent));
    }
  }
  assert(false);
  return nullptr;
}

void deserialize(Deserializer &ctx, FutureWrapper &value)
{
  DomainPoint point;
  deserialize(ctx, point);

  auto future  = ctx.futures_[0];
  ctx.futures_ = ctx.futures_.subspan(1);

  Domain domain;
  domain.dim = point.dim;
  for (int32_t idx = 0; idx < point.dim; ++idx) {
    domain.rect_data[idx]             = 0;
    domain.rect_data[idx + point.dim] = point[idx] - 1;
  }

  value = FutureWrapper(domain, future);
}

void deserialize(Deserializer &ctx, RegionField &value)
{
  auto dim = ctx.deserializer_.unpack_32bit_int();
  auto idx = ctx.deserializer_.unpack_32bit_uint();
  auto fid = ctx.deserializer_.unpack_32bit_int();

  auto &pr = ctx.regions_[idx];
  value    = RegionField(dim, pr, fid);
}

void deserialize(Deserializer &ctx, OutputRegionField &value)
{
  auto idx = ctx.deserializer_.unpack_32bit_uint();
  auto fid = ctx.deserializer_.unpack_32bit_int();

  auto &out = ctx.outputs_[idx];
  value     = OutputRegionField(out, fid);
}

void deserialize(Deserializer &ctx, Store &store)
{
  auto is_future = ctx.deserializer_.unpack_bool();
  auto dim       = ctx.deserializer_.unpack_32bit_int();
  auto code      = ctx.deserializer_.unpack_dtype();

  auto transform = deserialize_transform(ctx);
  auto redop_id  = ctx.deserializer_.unpack_32bit_int();

  if (is_future) {
    FutureWrapper fut;
    deserialize(ctx, fut);
    store = Store(dim, code, redop_id, fut, std::move(transform));
  } else if (dim >= 0) {
    RegionField rf;
    deserialize(ctx, rf);
    store = Store(dim, code, redop_id, std::move(rf), std::move(transform));
  } else {
    OutputRegionField out;
    deserialize(ctx, out);
    store = Store(code, std::move(out), std::move(transform));
  }
}

}  // namespace legate
