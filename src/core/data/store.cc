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

#include "core/data/buffer.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/machine.h"
#include "legate_defines.h"

#ifdef LEGATE_USE_CUDA
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#endif

namespace legate {

RegionField::RegionField(int32_t dim, const Legion::PhysicalRegion& pr, Legion::FieldID fid)
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

bool RegionField::valid() const
{
  return pr_.get_logical_region() != Legion::LogicalRegion::NO_REGION;
}

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

UnboundRegionField::UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid)
  : out_(out),
    fid_(fid),
    num_elements_(
      Legion::UntypedDeferredValue(sizeof(size_t), find_memory_kind_for_executing_processor()))
{
}

UnboundRegionField::UnboundRegionField(UnboundRegionField&& other) noexcept
  : bound_(other.bound_), out_(other.out_), fid_(other.fid_), num_elements_(other.num_elements_)
{
  other.bound_        = false;
  other.out_          = Legion::OutputRegion();
  other.fid_          = -1;
  other.num_elements_ = Legion::UntypedDeferredValue();
}

UnboundRegionField& UnboundRegionField::operator=(UnboundRegionField&& other) noexcept
{
  bound_        = other.bound_;
  out_          = other.out_;
  fid_          = other.fid_;
  num_elements_ = other.num_elements_;

  other.bound_        = false;
  other.out_          = Legion::OutputRegion();
  other.fid_          = -1;
  other.num_elements_ = Legion::UntypedDeferredValue();

  return *this;
}

void UnboundRegionField::bind_empty_data(int32_t ndim)
{
  update_num_elements(0);
  DomainPoint extents;
  extents.dim = ndim;
  for (int32_t dim = 0; dim < ndim; ++dim) extents[dim] = 0;
  auto empty_buffer = create_buffer<int8_t>(0);
  out_.return_data(extents, fid_, empty_buffer.get_instance(), false);
  bound_ = true;
}

ReturnValue UnboundRegionField::pack_weight() const
{
#ifdef DEBUG_LEGATE
  if (!bound_) {
    legate::log_legate.error(
      "Found an uninitialized unbound store. Please make sure you return buffers to all unbound "
      "stores in the task");
    LEGATE_ABORT;
  }
#endif
  return ReturnValue(num_elements_, sizeof(size_t));
}

void UnboundRegionField::update_num_elements(size_t num_elements)
{
  AccessorWO<size_t, 1> acc(num_elements_, sizeof(size_t), false);
  acc[0] = num_elements;
}

FutureWrapper::FutureWrapper(bool read_only,
                             uint32_t field_size,
                             Domain domain,
                             Legion::Future future,
                             bool initialize /*= false*/)
  : read_only_(read_only), field_size_(field_size), domain_(domain), future_(future)
{
#ifdef DEBUG_LEGATE
  assert(field_size > 0);
#endif
  if (!read_only) {
#ifdef DEBUG_LEGATE
    assert(!initialize || future_.get_untyped_size() == field_size);
#endif
    auto mem_kind = find_memory_kind_for_executing_processor(
#ifdef LEGATE_NO_FUTURES_ON_FB
      true
#else
      false
#endif
    );
    if (initialize) {
      auto p_init_value = future_.get_buffer(mem_kind);
#ifdef LEGATE_USE_CUDA
      if (mem_kind == Memory::Kind::GPU_FB_MEM) {
        // TODO: This should be done by Legion
        buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind);
        AccessorWO<int8_t, 1> acc(buffer_, field_size, false);
        auto stream = cuda::StreamPool::get_stream_pool().get_stream();
        CHECK_CUDA(
          cudaMemcpyAsync(acc.ptr(0), p_init_value, field_size, cudaMemcpyDeviceToDevice, stream));
      } else
#endif
        buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind, p_init_value);
    } else
      buffer_ = Legion::UntypedDeferredValue(field_size, mem_kind);
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

  auto redop = Legion::Runtime::get_reduction_op(redop_id);
#ifdef DEBUG_LEGATE
  assert(redop->sizeof_lhs == field_size_);
#endif
  auto identity = redop->identity;
#ifdef LEGATE_USE_CUDA
  if (buffer_.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();
    CHECK_CUDA(cudaMemcpyAsync(ptr, identity, field_size_, cudaMemcpyHostToDevice, stream));
  } else
#endif
    memcpy(ptr, identity, field_size_);
}

ReturnValue FutureWrapper::pack() const { return ReturnValue(buffer_, field_size_); }

Store::Store(int32_t dim,
             std::unique_ptr<Type> type,
             int32_t redop_id,
             FutureWrapper future,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(true),
    is_unbound_store_(false),
    dim_(dim),
    type_(std::move(type)),
    redop_id_(redop_id),
    future_(future),
    transform_(std::forward<decltype(transform)>(transform)),
    readable_(true)
{
}

Store::Store(int32_t dim,
             std::unique_ptr<Type> type,
             int32_t redop_id,
             RegionField&& region_field,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_unbound_store_(false),
    dim_(dim),
    type_(std::move(type)),
    redop_id_(redop_id),
    region_field_(std::forward<RegionField>(region_field)),
    transform_(std::forward<decltype(transform)>(transform))
{
  readable_  = region_field_.is_readable();
  writable_  = region_field_.is_writable();
  reducible_ = region_field_.is_reducible();
}

Store::Store(int32_t dim,
             std::unique_ptr<Type> type,
             UnboundRegionField&& unbound_field,
             std::shared_ptr<TransformStack>&& transform)
  : is_future_(false),
    is_unbound_store_(true),
    dim_(dim),
    type_(std::move(type)),
    redop_id_(-1),
    unbound_field_(std::forward<UnboundRegionField>(unbound_field)),
    transform_(std::forward<decltype(transform)>(transform))
{
}

Store::Store(Store&& other) noexcept
  : is_future_(other.is_future_),
    is_unbound_store_(other.is_unbound_store_),
    dim_(other.dim_),
    type_(std::move(other.type_)),
    redop_id_(other.redop_id_),
    future_(other.future_),
    region_field_(std::forward<RegionField>(other.region_field_)),
    unbound_field_(std::forward<UnboundRegionField>(other.unbound_field_)),
    transform_(std::move(other.transform_)),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

Store& Store::operator=(Store&& other) noexcept
{
  is_future_        = other.is_future_;
  is_unbound_store_ = other.is_unbound_store_;
  dim_              = other.dim_;
  type_             = std::move(other.type_);
  redop_id_         = other.redop_id_;
  if (is_future_)
    future_ = other.future_;
  else if (is_unbound_store_)
    unbound_field_ = std::move(other.unbound_field_);
  else
    region_field_ = std::move(other.region_field_);
  transform_ = std::move(other.transform_);
  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

bool Store::valid() const { return is_future_ || is_unbound_store_ || region_field_.valid(); }

Domain Store::domain() const
{
#ifdef DEBUG_LEGATE
  assert(!is_unbound_store_);
#endif
  auto result = is_future_ ? future_.domain() : region_field_.domain();
  if (!transform_->identity()) result = transform_->transform(result);
#ifdef DEBUG_LEGATE
  assert(result.dim == dim_ || dim_ == 0);
#endif
  return result;
}

void Store::bind_empty_data()
{
#ifdef DEBUG_LEGATE
  check_valid_return();
#endif
  unbound_field_.bind_empty_data(dim_);
}

void Store::remove_transform()
{
#ifdef DEBUG_LEGATE
  assert(transformed());
#endif
  dim_ = transform_->pop()->target_ndim(dim_);
}

void Store::check_valid_return() const
{
  if (!is_unbound_store_) {
    log_legate.error("Invalid to return a buffer to a bound store");
    LEGATE_ABORT;
  }
  if (unbound_field_.bound()) {
    log_legate.error("Invalid to return more than one buffer to an unbound store");
    LEGATE_ABORT;
  }
}

void Store::check_buffer_dimension(const int32_t dim) const
{
  if (dim != dim_) {
    log_legate.error(
      "Dimension mismatch: invalid to bind a %d-D buffer to a %d-D store", dim, dim_);
    LEGATE_ABORT;
  }
}

void Store::check_accessor_dimension(const int32_t dim) const
{
  if (!(dim == dim_ || (dim_ == 0 && dim == 1))) {
    log_legate.error(
      "Dimension mismatch: invalid to create a %d-D accessor to a %d-D store", dim, dim_);
    LEGATE_ABORT;
  }
}

}  // namespace legate
