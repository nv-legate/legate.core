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

UntypedPoint::~UntypedPoint() { destroy(); }

UntypedPoint::UntypedPoint(const UntypedPoint &other)
{
  destroy();
  copy(other);
}

UntypedPoint &UntypedPoint::operator=(const UntypedPoint &other)
{
  destroy();
  copy(other);
  return *this;
}

UntypedPoint::UntypedPoint(UntypedPoint &&other) noexcept
{
  destroy();
  move(std::forward<UntypedPoint>(other));
}

UntypedPoint &UntypedPoint::operator=(UntypedPoint &&other) noexcept
{
  destroy();
  move(std::forward<UntypedPoint>(other));
  return *this;
}

void UntypedPoint::copy(const UntypedPoint &other)
{
  if (exists()) {
    N_     = other.N_;
    point_ = dim_dispatch(N_, copy_fn{}, other.point_);
  }
}

void UntypedPoint::move(UntypedPoint &&other)
{
  N_     = other.N_;
  point_ = other.point_;

  other.N_     = -1;
  other.point_ = nullptr;
}

void UntypedPoint::destroy()
{
  if (exists()) {
    dim_dispatch(N_, destroy_fn{}, point_);
    N_ = -1;
  }
}

struct point_to_ostream_fn {
  template <int32_t N>
  void operator()(std::ostream &os, const UntypedPoint &point)
  {
    os << point.to_point<N>();
  }
};

std::ostream &operator<<(std::ostream &os, const UntypedPoint &point)
{
  dim_dispatch(point.dim(), point_to_ostream_fn{}, os, point);
  return os;
}

Shape::~Shape() { destroy(); }

Shape::Shape(const Shape &other)
{
  destroy();
  copy(other);
}

Shape &Shape::operator=(const Shape &other)
{
  destroy();
  copy(other);
  return *this;
}

Shape::Shape(Shape &&other) noexcept
{
  destroy();
  move(std::forward<Shape>(other));
}

Shape &Shape::operator=(Shape &&other) noexcept
{
  destroy();
  move(std::forward<Shape>(other));
  return *this;
}

void Shape::copy(const Shape &other)
{
  if (exists()) {
    N_    = other.N_;
    rect_ = dim_dispatch(N_, copy_fn{}, other.rect_);
  }
}

void Shape::move(Shape &&other)
{
  N_    = other.N_;
  rect_ = other.rect_;

  other.N_    = -1;
  other.rect_ = nullptr;
}

void Shape::destroy()
{
  if (exists()) {
    dim_dispatch(N_, destroy_fn{}, rect_);
    N_ = -1;
  }
}

struct shape_to_ostream_fn {
  template <int32_t N>
  void operator()(std::ostream &os, const Shape &shape)
  {
    os << shape.to_rect<N>();
  }
};

std::ostream &operator<<(std::ostream &os, const Shape &shape)
{
  dim_dispatch(shape.dim(), shape_to_ostream_fn{}, os, shape);
  return os;
}

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
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t i = 0, j = 0; i < out_dim; ++i)
    if (i != dim_) transform.matrix[i * in_dim + j++] = 1;

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

RegionField::RegionField(int32_t dim, const PhysicalRegion &pr, FieldID fid)
  : dim_(dim), pr_(pr), fid_(fid)
{
}

RegionField::RegionField(RegionField &&other) noexcept
  : dim_(other.dim_), pr_(other.pr_), fid_(other.fid_)
{
}

RegionField &RegionField::operator=(RegionField &&other) noexcept
{
  dim_ = other.dim_;
  pr_  = other.pr_;
  fid_ = other.fid_;
  return *this;
}

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

Store::Store(int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             Future future,
             std::unique_ptr<Transform> transform)
  : is_future_(true),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    future_(future),
    transform_(std::move(transform))
{
}

Store::Store(int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             RegionField &&region_field,
             std::unique_ptr<Transform> transform)
  : is_future_(false),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(std::forward<RegionField>(region_field)),
    transform_(std::move(transform))
{
}

Store &Store::operator=(Store &&other) noexcept
{
  is_future_ = other.is_future_;
  dim_       = other.dim_;
  code_      = other.code_;
  redop_id_  = other.redop_id_;
  if (is_future_)
    future_ = other.future_;
  else
    region_field_ = std::move(other.region_field_);
  transform_ = std::move(other.transform_);
  return *this;
}

Domain Store::domain() const
{
  auto result = is_future_ ? Domain(Rect<1>(Point<1>(0), Point<1>(0))) : region_field_.domain();
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

}  // namespace legate
