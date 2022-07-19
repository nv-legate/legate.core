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

#include "core/data/store.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/machine.h"

namespace legate {

using namespace Legion;

RegionField::RegionField(int32_t dim, const PhysicalRegion& pr, FieldID fid)
  : dim_(dim), pr_(pr), fid_(fid)
{
  auto priv  = pr.get_privilege();
  readable_  = static_cast<bool>(priv & LEGION_READ_PRIV);
  writable_  = static_cast<bool>(priv & LEGION_WRITE_PRIV);
  reducible_ = static_cast<bool>(priv & LEGION_REDUCE) || (readable_ && writable_);
}

RegionField::RegionField(RegionField&& other) noexcept
  : dim_(other.dim_),
    pr_(other.pr_),
    fid_(other.fid_),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

RegionField& RegionField::operator=(RegionField&& other) noexcept
{
  dim_       = other.dim_;
  pr_        = other.pr_;
  fid_       = other.fid_;
  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

bool RegionField::valid() const { return pr_.get_logical_region() != LogicalRegion::NO_REGION; }

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

OutputRegionField::OutputRegionField(const OutputRegion& out, FieldID fid)
  : out_(out),
    fid_(fid),
    num_elements_(
      DeferredBuffer<size_t, 1>(Rect<1>(0, 0), find_memory_kind_for_executing_processor()))
{
}

OutputRegionField::OutputRegionField(OutputRegionField&& other) noexcept
  : bound_(other.bound_), out_(other.out_), fid_(other.fid_), num_elements_(other.num_elements_)
{
  other.bound_        = false;
  other.out_          = OutputRegion();
  other.fid_          = -1;
  other.num_elements_ = DeferredBuffer<size_t, 1>();
}

OutputRegionField& OutputRegionField::operator=(OutputRegionField&& other) noexcept
{
  bound_        = other.bound_;
  out_          = other.out_;
  fid_          = other.fid_;
  num_elements_ = other.num_elements_;

  other.bound_        = false;
  other.out_          = OutputRegion();
  other.fid_          = -1;
  other.num_elements_ = DeferredBuffer<size_t, 1>();

  return *this;
}

void OutputRegionField::make_empty(int32_t ndim)
{
  num_elements_[0] = 0;
  DomainPoint extents;
  extents.dim = ndim;
  for (int32_t dim = 0; dim < ndim; ++dim) extents[dim] = 0;
  out_.return_data(extents, fid_, nullptr);
}

ReturnValue OutputRegionField::pack_weight() const
{
  return ReturnValue(num_elements_.ptr(0), sizeof(size_t));
}

FutureWrapper::FutureWrapper(
  bool read_only, int32_t field_size, Domain domain, Future future, bool initialize /*= false*/)
  : read_only_(read_only), field_size_(field_size), domain_(domain), future_(future)
{
  assert(field_size > 0);
  if (!read_only) {
    auto mem_kind = find_memory_kind_for_executing_processor();
    assert(!initialize || future_.get_untyped_size() == field_size);
    auto p_init_value = initialize ? future_.get_buffer(mem_kind) : nullptr;
    buffer_           = UntypedDeferredValue(field_size, mem_kind, p_init_value);
  }
}

FutureWrapper::FutureWrapper(const FutureWrapper& other) noexcept
  : read_only_(other.read_only_),
    field_size_(other.field_size_),
    domain_(other.domain_),
    future_(other.future_),
    buffer_(other.buffer_)
{
}

FutureWrapper& FutureWrapper::operator=(const FutureWrapper& other) noexcept
{
  read_only_  = other.read_only_;
  field_size_ = other.field_size_;
  domain_     = other.domain_;
  future_     = other.future_;
  buffer_     = other.buffer_;
  return *this;
}

Domain FutureWrapper::domain() const { return domain_; }

void FutureWrapper::initialize_with_identity(int32_t redop_id)
{
  auto untyped_acc = AccessorWO<int8_t, 1>(buffer_, field_size_);
  auto ptr         = untyped_acc.ptr(0);

  auto redop = Runtime::get_reduction_op(redop_id);
  assert(redop->sizeof_lhs == field_size_);
  auto identity = redop->identity;
  memcpy(ptr, identity, field_size_);
}

ReturnValue FutureWrapper::pack() const
{
  auto untyped_acc = AccessorRO<int8_t, 1>(buffer_, field_size_);
  auto ptr         = untyped_acc.ptr(0);
  return ReturnValue(ptr, field_size_);
}

Store::Store(int32_t dim,
             int32_t code,
             int32_t redop_id,
             FutureWrapper future,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(true),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    future_(future),
    transform_(std::forward<decltype(transform)>(transform)),
    readable_(true)
{
}

Store::Store(int32_t dim,
             int32_t code,
             int32_t redop_id,
             RegionField&& region_field,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(std::forward<RegionField>(region_field)),
    transform_(std::forward<decltype(transform)>(transform))
{
  readable_  = region_field_.is_readable();
  writable_  = region_field_.is_writable();
  reducible_ = region_field_.is_reducible();
}

Store::Store(int32_t dim,
             int32_t code,
             OutputRegionField&& output,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_output_store_(true),
    dim_(dim),
    code_(code),
    redop_id_(-1),
    output_field_(std::forward<OutputRegionField>(output)),
    transform_(std::forward<decltype(transform)>(transform))
{
}

Store::Store(Store&& other) noexcept
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

Store& Store::operator=(Store&& other) noexcept
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

bool Store::valid() const { return is_future_ || is_output_store_ || region_field_.valid(); }

Domain Store::domain() const
{
  assert(!is_output_store_);
  auto result = is_future_ ? future_.domain() : region_field_.domain();
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

void Store::make_empty()
{
  assert(is_output_store_);
  output_field_.make_empty(dim_);
}

void Store::remove_transform()
{
  assert(is_transformed());
  auto result = transform_->pop();
  if (transform_->empty()) transform_ = nullptr;
  dim_ = result->target_ndim(dim_);
}

}  // namespace legate
