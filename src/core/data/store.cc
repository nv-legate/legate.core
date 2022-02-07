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

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

OutputRegionField::OutputRegionField(const OutputRegion& out, FieldID fid) : out_(out), fid_(fid) {}

OutputRegionField::OutputRegionField(OutputRegionField&& other) noexcept
  : bound_(other.bound_), out_(other.out_), fid_(other.fid_)
{
  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;
}

OutputRegionField& OutputRegionField::operator=(OutputRegionField&& other) noexcept
{
  bound_ = other.bound_;
  out_   = other.out_;
  fid_   = other.fid_;

  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;

  return *this;
}

FutureWrapper::FutureWrapper(
  bool read_only, int32_t field_size, Domain domain, Future future, bool initialize /*= false*/)
  : read_only_(read_only),
    field_size_(field_size),
    domain_(domain),
    future_(future),
    uninitialized_(!initialize)
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
    buffer_(other.buffer_),
    uninitialized_(other.uninitialized_),
    rawptr_(other.rawptr_)
{
}

FutureWrapper& FutureWrapper::operator=(const FutureWrapper& other) noexcept
{
  read_only_     = other.read_only_;
  field_size_    = other.field_size_;
  domain_        = other.domain_;
  future_        = other.future_;
  buffer_        = other.buffer_;
  uninitialized_ = other.uninitialized_;
  rawptr_        = other.rawptr_;
  return *this;
}

Domain FutureWrapper::domain() const { return domain_; }

ReturnValue FutureWrapper::pack() const
{
  if (nullptr == rawptr_) {
    fprintf(stderr, "Found an uninitialized Legate store\n");
    assert(false);
  }
  return ReturnValue(rawptr_, field_size_);
}

Store::Store(int32_t dim,
             LegateTypeCode code,
             FutureWrapper future,
             std::shared_ptr<StoreTransform> transform)
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
             RegionField&& region_field,
             std::shared_ptr<StoreTransform> transform)
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

Store::Store(LegateTypeCode code,
             OutputRegionField&& output,
             std::shared_ptr<StoreTransform> transform)
  : is_future_(false),
    is_output_store_(true),
    dim_(-1),
    code_(code),
    redop_id_(-1),
    output_field_(std::forward<OutputRegionField>(output)),
    transform_(std::move(transform))
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

Domain Store::domain() const
{
  assert(!is_output_store_);
  auto result = is_future_ ? future_.domain() : region_field_.domain();
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

}  // namespace legate
