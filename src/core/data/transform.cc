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

#include "core/data/transform.h"

namespace legate {

Legion::DomainAffineTransform combine(const Legion::DomainAffineTransform& lhs,
                                      const Legion::DomainAffineTransform& rhs)
{
  Legion::DomainAffineTransform result;
  auto transform   = lhs.transform * rhs.transform;
  auto offset      = lhs.transform * rhs.offset + lhs.offset;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

TransformStack::TransformStack(std::unique_ptr<StoreTransform>&& transform,
                               std::shared_ptr<TransformStack>&& parent)
  : transform_(std::forward<decltype(transform)>(transform)),
    parent_(std::forward<decltype(parent)>(parent))
{
}

Domain TransformStack::transform(const Domain& input) const
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);
#endif
  return transform_->transform(parent_->identity() ? input : parent_->transform(input));
}

Legion::DomainAffineTransform TransformStack::inverse_transform(int32_t in_dim) const
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);
#endif
  auto result  = transform_->inverse_transform(in_dim);
  auto out_dim = transform_->target_ndim(in_dim);

  if (parent_->identity())
    return result;
  else {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  }
}

void TransformStack::print(std::ostream& out) const
{
  if (identity()) {
    out << "(identity)";
    return;
  }

  transform_->print(out);
  if (!parent_->identity()) {
    out << " >> ";
    parent_->print(out);
  }
}

std::unique_ptr<StoreTransform> TransformStack::pop()
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);
#endif
  auto result = std::move(transform_);
  if (parent_ != nullptr) {
    transform_ = std::move(parent_->transform_);
    parent_    = std::move(parent_->parent_);
  }
  return std::move(result);
}

void TransformStack::dump() const { std::cerr << *this << std::endl; }

std::vector<int32_t> TransformStack::find_imaginary_dims() const
{
  std::vector<int32_t> dims;
  if (nullptr != parent_) { dims = parent_->find_imaginary_dims(); }
  if (nullptr != transform_) transform_->find_imaginary_dims(dims);
  return std::move(dims);
}

Shift::Shift(int32_t dim, int64_t offset) : dim_(dim), offset_(offset) {}

Domain Shift::transform(const Domain& input) const
{
  auto result = input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

Legion::DomainAffineTransform Shift::inverse_transform(int32_t in_dim) const
{
  assert(dim_ < in_dim);
  auto out_dim = in_dim;

  Legion::DomainTransform transform;
  transform.m = out_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j)
      transform.matrix[i * in_dim + j] = static_cast<coord_t>(i == j);

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = i == dim_ ? -offset_ : 0;

  Legion::DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

void Shift::print(std::ostream& out) const
{
  out << "Shift(";
  out << "dim: " << dim_ << ", ";
  out << "offset: " << offset_ << ")";
}

int32_t Shift::target_ndim(int32_t source_ndim) const { return source_ndim; }

void Shift::find_imaginary_dims(std::vector<int32_t>&) const {}

Promote::Promote(int32_t extra_dim, int64_t dim_size) : extra_dim_(extra_dim), dim_size_(dim_size)
{
}

Domain Promote::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim + 1;

  for (int32_t out_dim = 0, in_dim = 0; out_dim < output.dim; ++out_dim)
    if (out_dim == extra_dim_) {
      output.rect_data[out_dim]              = 0;
      output.rect_data[out_dim + output.dim] = dim_size_ - 1;
    } else {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
      ++in_dim;
    }
  return output;
}

Legion::DomainAffineTransform Promote::inverse_transform(int32_t in_dim) const
{
  assert(extra_dim_ < in_dim);
  auto out_dim = in_dim - 1;

  Legion::DomainTransform transform;
  transform.m = std::max<int32_t>(out_dim, 1);
  transform.n = in_dim;
  for (int32_t i = 0; i < transform.m; ++i)
    for (int32_t j = 0; j < transform.n; ++j) transform.matrix[i * in_dim + j] = 0;

  if (out_dim > 0)
    for (int32_t j = 0, i = 0; j < transform.n; ++j)
      if (j != extra_dim_) transform.matrix[i++ * in_dim + j] = 1;

  DomainPoint offset;
  offset.dim = std::max<int32_t>(out_dim, 1);
  for (int32_t i = 0; i < transform.m; ++i) offset[i] = 0;

  Legion::DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

void Promote::print(std::ostream& out) const
{
  out << "Promote(";
  out << "extra_dim: " << extra_dim_ << ", ";
  out << "dim_size: " << dim_size_ << ")";
}

int32_t Promote::target_ndim(int32_t source_ndim) const { return source_ndim - 1; }

void Promote::find_imaginary_dims(std::vector<int32_t>& dims) const
{
  for (auto& dim : dims)
    if (dim >= extra_dim_) dim++;
  dims.push_back(extra_dim_);
}

Project::Project(int32_t dim, int64_t coord) : dim_(dim), coord_(coord) {}

Domain Project::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim - 1;

  for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim)
    if (in_dim != dim_) {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
      ++out_dim;
    }
  return output;
}

Legion::DomainAffineTransform Project::inverse_transform(int32_t in_dim) const
{
  auto out_dim = in_dim + 1;
  assert(dim_ < out_dim);

  Legion::DomainTransform transform;
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

  Legion::DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

void Project::print(std::ostream& out) const
{
  out << "Project(";
  out << "dim: " << dim_ << ", ";
  out << "coord: " << coord_ << ")";
}

int32_t Project::target_ndim(int32_t source_ndim) const { return source_ndim + 1; }

void Project::find_imaginary_dims(std::vector<int32_t>& dims) const
{
  auto finder = std::find(dims.begin(), dims.end(), dim_);
  if (finder != dims.end()) { dims.erase(finder); }
  for (auto& dim : dims)
    if (dim > dim_) --dim;
}

Transpose::Transpose(std::vector<int32_t>&& axes) : axes_(std::move(axes)) {}

Domain Transpose::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim;
  for (int32_t out_dim = 0; out_dim < output.dim; ++out_dim) {
    auto in_dim                            = axes_[out_dim];
    output.rect_data[out_dim]              = input.rect_data[in_dim];
    output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
  }
  return output;
}

Legion::DomainAffineTransform Transpose::inverse_transform(int32_t in_dim) const
{
  Legion::DomainTransform transform;
  transform.m = in_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < in_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t j = 0; j < in_dim; ++j) transform.matrix[axes_[j] * in_dim + j] = 1;

  DomainPoint offset;
  offset.dim = in_dim;
  for (int32_t i = 0; i < in_dim; ++i) offset[i] = 0;

  Legion::DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

namespace {  // anonymous
template <typename T>
void print_vector(std::ostream& out, const std::vector<T>& vec)
{
  bool past_first = false;
  out << "[";
  for (const T& val : vec) {
    if (past_first) {
      out << ", ";
    } else {
      past_first = true;
    }
    out << val;
  }
  out << "]";
}
}  // anonymous namespace

void Transpose::print(std::ostream& out) const
{
  out << "Transpose(";
  out << "axes: ";
  print_vector(out, axes_);
  out << ")";
}

int32_t Transpose::target_ndim(int32_t source_ndim) const { return source_ndim; }

void Transpose::find_imaginary_dims(std::vector<int32_t>& dims) const
{
  // i should be added to X.tranpose(axes).promoted iff axes[i] is in X.promoted
  // e.g. X.promoted = [0] => X.transpose((1,2,0)).promoted = [2]
  for (auto& promoted : dims) {
    auto finder = std::find(axes_.begin(), axes_.end(), promoted);
#ifdef DEBUG_LEGATE
    assert(finder != axes_.end());
#endif
    promoted = finder - axes_.begin();
  }
}

Delinearize::Delinearize(int32_t dim, std::vector<int64_t>&& sizes)
  : dim_(dim), sizes_(std::move(sizes)), strides_(sizes_.size(), 1), volume_(1)
{
  for (int32_t dim = sizes_.size() - 2; dim >= 0; --dim)
    strides_[dim] = strides_[dim + 1] * sizes_[dim + 1];
  for (auto size : sizes_) volume_ *= size;
}

Domain Delinearize::transform(const Domain& input) const
{
  auto delinearize = [](const auto dim, const auto ndim, const auto& strides, const Domain& input) {
    Domain output;
    output.dim     = input.dim - 1 + ndim;
    int32_t in_dim = 0;
    for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim) {
      if (in_dim == dim) {
        auto lo = input.rect_data[in_dim];
        auto hi = input.rect_data[input.dim + in_dim];
        for (auto stride : strides) {
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
  };
  return delinearize(dim_, sizes_.size(), strides_, input);
}

Legion::DomainAffineTransform Delinearize::inverse_transform(int32_t in_dim) const
{
  Legion::DomainTransform transform;
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

  Legion::DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

void Delinearize::print(std::ostream& out) const
{
  out << "Delinearize(";
  out << "dim: " << dim_ << ", ";
  out << "sizes: ";
  print_vector(out, sizes_);
  out << ")";
}

int32_t Delinearize::target_ndim(int32_t source_ndim) const
{
  return source_ndim - strides_.size() + 1;
}

void Delinearize::find_imaginary_dims(std::vector<int32_t>&) const {}

std::ostream& operator<<(std::ostream& out, const Transform& transform)
{
  transform.print(out);
  return out;
}

}  // namespace legate
