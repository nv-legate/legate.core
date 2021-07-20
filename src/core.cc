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

#include "core.h"
#include "dispatch.h"

namespace legate {

using namespace Legion;

using TransformP = std::unique_ptr<Transform>;

DomainTransform operator*(const DomainTransform &lhs, const DomainTransform &rhs)
{
  assert(lhs.n == rhs.m);
  DomainTransform result;
  result.m = lhs.m;
  result.n = rhs.n;
  for (int32_t i = 0; i < result.m; ++i)
    for (int32_t j = 0; j < result.n; ++j) {
      result.matrix[i * result.n + j] = 0;
      for (int32_t k = 0; k < lhs.n; ++k)
        result.matrix[i * result.n + j] += lhs.matrix[i * lhs.n + k] * rhs.matrix[k * rhs.n + j];
    }
  return result;
}

DomainPoint operator+(const DomainPoint &lhs, const DomainPoint &rhs)
{
  assert(lhs.dim == rhs.dim);
  DomainPoint result(lhs);
  for (int32_t idx = 0; idx < rhs.dim; ++idx) result[idx] += rhs[idx];
  return result;
}

DomainAffineTransform combine(const DomainAffineTransform &lhs, const DomainAffineTransform &rhs)
{
  DomainAffineTransform result;
  auto transform   = lhs.transform * rhs.transform;
  auto offset      = lhs.transform * rhs.offset + lhs.offset;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

Transform::Transform(TransformP &&parent) : parent_(std::move(parent)) {}

Shift::Shift(int32_t dim, int64_t offset, TransformP &&parent)
  : Transform(std::forward<TransformP>(parent)), dim_(dim), offset_(offset)
{
}

Domain Shift::transform(const Domain &input) const
{
  auto result = nullptr != parent_ ? parent_->transform(input) : input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

DomainAffineTransform Shift::inverse_transform(int32_t in_dim) const
{
  assert(dim_ < in_dim);
  auto out_dim = in_dim;

  DomainTransform transform;
  transform.m = out_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j)
      transform.matrix[i * in_dim + j] = static_cast<coord_t>(i == j);

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = i == dim_ ? -offset_ : 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Promote::Promote(int32_t extra_dim, int64_t dim_size, TransformP &&parent)
  : Transform(std::forward<TransformP>(parent)), extra_dim_(extra_dim), dim_size_(dim_size)
{
}

Domain Promote::transform(const Domain &input) const
{
  auto promote = [](int32_t extra_dim, int64_t dim_size, const Domain &input) {
    Domain output;
    output.dim = input.dim + 1;

    for (int32_t out_dim = 0, in_dim = 0; out_dim < output.dim; ++out_dim)
      if (out_dim == extra_dim) {
        output.rect_data[out_dim]              = 0;
        output.rect_data[out_dim + output.dim] = dim_size - 1;
      } else {
        output.rect_data[out_dim]              = input.rect_data[in_dim];
        output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
        ++in_dim;
      }
    return output;
  };

  return promote(extra_dim_, dim_size_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Promote::inverse_transform(int32_t in_dim) const
{
  assert(extra_dim_ < in_dim);
  auto out_dim = in_dim - 1;

  DomainTransform transform;
  transform.m = out_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t j = 0, i = 0; j < in_dim; ++j)
    if (j != extra_dim_) transform.matrix[i++ * in_dim + j] = 1;

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Project::Project(int32_t dim, int64_t coord, TransformP &&parent)
  : Transform(std::forward<TransformP>(parent)), dim_(dim), coord_(coord)
{
}

Domain Project::transform(const Domain &input) const
{
  auto project = [](int32_t collapsed_dim, const Domain &input) {
    Domain output;
    output.dim = input.dim - 1;

    for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim)
      if (in_dim != collapsed_dim) {
        output.rect_data[out_dim]              = input.rect_data[in_dim];
        output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
        ++out_dim;
      }
    return output;
  };

  return project(dim_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Project::inverse_transform(int32_t in_dim) const
{
  auto out_dim = in_dim + 1;
  assert(dim_ < out_dim);

  DomainTransform transform;
  transform.m = out_dim;
  if (in_dim == 0) {
    transform.n         = out_dim;
    transform.matrix[0] = 0;
  } else {
    transform.n = in_dim;
    for (int32_t i = 0; i < out_dim; ++i)
      for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

    for (int32_t i = 0, j = 0; i < out_dim; ++i)
      if (i != dim_) transform.matrix[i * in_dim + j++] = 1;
  }

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = i == dim_ ? coord_ : 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Transpose::Transpose(std::vector<int32_t> &&axes, TransformP &&parent)
  : Transform(std::forward<TransformP>(parent)), axes_(std::move(axes))
{
}

Domain Transpose::transform(const Domain &input) const
{
  auto transpose = [](const auto &axes, const Domain &input) {
    Domain output;
    output.dim = input.dim;
    for (int32_t in_dim = 0; in_dim < input.dim; ++in_dim) {
      auto out_dim                           = axes[in_dim];
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
    }
    return output;
  };

  return transpose(axes_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Transpose::inverse_transform(int32_t in_dim) const
{
  DomainTransform transform;
  transform.m = in_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < in_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t i = 0, j = 0; i < in_dim; ++i) transform.matrix[i * in_dim + axes_[i]] = 1;

  DomainPoint offset;
  offset.dim = in_dim;
  for (int32_t i = 0; i < in_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(in_dim);
    return combine(parent, result);
  } else
    return result;
}

Delinearize::Delinearize(int32_t dim, std::vector<int64_t> &&sizes, TransformP &&parent)
  : Transform(std::forward<TransformP>(parent)),
    dim_(dim),
    sizes_(std::move(sizes)),
    strides_(sizes_.size(), 1),
    volume_(1)
{
  for (int32_t dim = sizes_.size() - 2; dim >= 0; --dim)
    strides_[dim] = strides_[dim + 1] * sizes_[dim + 1];
  for (auto size : sizes_) volume_ *= size;
}

Domain Delinearize::transform(const Domain &input) const
{
  Domain output;
  output.dim     = input.dim - 1 + sizes_.size();
  int32_t in_dim = 0;
  for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim) {
    if (in_dim == dim_) {
      auto lo = input.rect_data[in_dim];
      auto hi = input.rect_data[input.dim + in_dim];
      for (auto stride : strides_) {
        output.rect_data[out_dim]              = lo / stride;
        output.rect_data[output.dim + out_dim] = hi / stride;
        lo                                     = lo % stride;
        hi                                     = hi % stride;
        ++out_dim;
      }
    } else {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[output.dim + out_dim] = input.rect_data[input.dim + in_dim];
      ++out_dim;
    }
  }
  return output;
}

DomainAffineTransform Delinearize::inverse_transform(int32_t in_dim) const
{
  DomainTransform transform;
  int32_t out_dim = in_dim - strides_.size() + 1;
  transform.m     = out_dim;
  transform.n     = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t i = 0, j = 0; i < out_dim; ++i)
    if (i == dim_)
      for (auto stride : strides_) transform.matrix[i * in_dim + j++] = stride;
    else
      transform.matrix[i * in_dim + j++] = 1;

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

RegionField::RegionField(int32_t dim, const PhysicalRegion &pr, FieldID fid)
  : dim_(dim), pr_(pr), fid_(fid)
{
  auto priv  = pr.get_privilege();
  readable_  = static_cast<bool>(priv & LEGION_READ_PRIV);
  writable_  = static_cast<bool>(priv & LEGION_WRITE_PRIV);
  reducible_ = static_cast<bool>(priv & LEGION_REDUCE) || (readable_ && writable_);
}

RegionField::RegionField(RegionField &&other) noexcept
  : dim_(other.dim_),
    pr_(other.pr_),
    fid_(other.fid_),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

RegionField &RegionField::operator=(RegionField &&other) noexcept
{
  dim_       = other.dim_;
  pr_        = other.pr_;
  fid_       = other.fid_;
  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

OutputRegionField::OutputRegionField(const OutputRegion &out, FieldID fid) : out_(out), fid_(fid) {}

OutputRegionField::OutputRegionField(OutputRegionField &&other) noexcept
  : bound_(other.bound_), out_(other.out_), fid_(other.fid_)
{
  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;
}

OutputRegionField &OutputRegionField::operator=(OutputRegionField &&other) noexcept
{
  bound_ = other.bound_;
  out_   = other.out_;
  fid_   = other.fid_;

  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;

  return *this;
}

FutureWrapper::FutureWrapper(Domain domain, Future future) : domain_(domain), future_(future) {}

FutureWrapper::FutureWrapper(const FutureWrapper &other) noexcept
  : domain_(other.domain_), future_(other.future_)
{
}

FutureWrapper &FutureWrapper::operator=(const FutureWrapper &other) noexcept
{
  domain_ = other.domain_;
  future_ = other.future_;
  return *this;
}

Domain FutureWrapper::domain() const { return domain_; }

Store::Store(int32_t dim,
             LegateTypeCode code,
             FutureWrapper future,
             std::unique_ptr<Transform> transform)
  : is_future_(true),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(-1),
    future_(future),
    transform_(std::move(transform)),
    readable_(true)
{
}

Store::Store(int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             RegionField &&region_field,
             std::unique_ptr<Transform> transform)
  : is_future_(false),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(std::forward<RegionField>(region_field)),
    transform_(std::move(transform))
{
  readable_  = region_field_.is_readable();
  writable_  = region_field_.is_writable();
  reducible_ = region_field_.is_reducible();
}

Store::Store(LegateTypeCode code, OutputRegionField &&output, std::unique_ptr<Transform> transform)
  : is_future_(false),
    is_output_store_(true),
    dim_(-1),
    code_(code),
    redop_id_(-1),
    output_field_(std::forward<OutputRegionField>(output)),
    transform_(std::move(transform))
{
}

Store::Store(Store &&other) noexcept
  : is_future_(other.is_future_),
    is_output_store_(other.is_output_store_),
    dim_(other.dim_),
    code_(other.code_),
    redop_id_(other.redop_id_),
    future_(other.future_),
    region_field_(std::forward<RegionField>(other.region_field_)),
    output_field_(std::forward<OutputRegionField>(other.output_field_)),
    transform_(std::move(other.transform_)),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

Store &Store::operator=(Store &&other) noexcept
{
  is_future_       = other.is_future_;
  is_output_store_ = other.is_output_store_;
  dim_             = other.dim_;
  code_            = other.code_;
  redop_id_        = other.redop_id_;
  if (is_future_)
    future_ = other.future_;
  else if (is_output_store_)
    output_field_ = std::move(other.output_field_);
  else
    region_field_ = std::move(other.region_field_);
  transform_ = std::move(other.transform_);
  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

Domain Store::domain() const
{
  assert(!is_output_store_);
  auto result = is_future_ ? future_.domain() : region_field_.domain();
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

}  // namespace legate
