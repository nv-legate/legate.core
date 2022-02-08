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

namespace legate {

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor() const
{
  return AccessorRO<T, DIM>(pr_, fid_);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor() const
{
  return AccessorWO<T, DIM>(pr_, fid_);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor() const
{
  return AccessorRW<T, DIM>(pr_, fid_);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(int32_t redop_id) const
{
  return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, redop_id);
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRO<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  int32_t redop_id, const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, redop_id, transform);
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::Rect<DIM>& bounds) const
{
  return AccessorRO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::Rect<DIM>& bounds) const
{
  return AccessorWO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(const Legion::Rect<DIM>& bounds) const
{
  return AccessorRW<T, DIM>(pr_, fid_, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(int32_t redop_id,
                                                            const Legion::Rect<DIM>& bounds) const
{
  return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, redop_id, bounds);
}

template <typename T, int32_t DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::Rect<DIM>& bounds,
                                              const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::Rect<DIM>& bounds,
                                               const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Legion::Rect<DIM>& bounds, const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  int32_t redop_id,
  const Legion::Rect<DIM>& bounds,
  const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, redop_id, transform, bounds);
}

template <int32_t DIM>
Legion::Rect<DIM> RegionField::shape() const
{
  return Legion::Rect<DIM>(pr_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> FutureWrapper::read_accessor() const
{
  assert(sizeof(T) == field_size_);
  if (read_only_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, memkind);
  } else
    return AccessorRO<T, DIM>(buffer_);
}

template <typename T, int DIM>
AccessorWO<T, DIM> FutureWrapper::write_accessor() const
{
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
  auto acc = AccessorWO<T, DIM>(buffer_);
  if (nullptr == rawptr_) rawptr_ = acc.ptr(Legion::Point<DIM>::ZEROES());
  return acc;
}

template <typename T, int DIM>
AccessorRW<T, DIM> FutureWrapper::read_write_accessor() const
{
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
  auto acc = AccessorRW<T, DIM>(buffer_);
  if (nullptr == rawptr_) rawptr_ = acc.ptr(Legion::Point<DIM>::ZEROES());
  return acc;
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> FutureWrapper::reduce_accessor(int32_t redop_id) const
{
  assert(sizeof(typename OP::LHS) == field_size_);
  assert(!read_only_);
  auto acc = AccessorRD<OP, EXCLUSIVE, DIM>(buffer_);
  if (nullptr == rawptr_) {
    auto p  = Legion::Point<DIM>::ZEROES();
    rawptr_ = acc.ptr(p);

    if (uninitialized_) {
      auto identity = OP::identity;
      memcpy(rawptr_, &identity, field_size_);
      uninitialized_ = false;
    }
  }
  return acc;
}

template <typename T, int DIM>
AccessorRO<T, DIM> FutureWrapper::read_accessor(const Legion::Rect<DIM>& bounds) const
{
  assert(sizeof(T) == field_size_);
  if (read_only_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, bounds, memkind);
  } else
    return AccessorRO<T, DIM>(buffer_, bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> FutureWrapper::write_accessor(const Legion::Rect<DIM>& bounds) const
{
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
  auto acc = AccessorWO<T, DIM>(buffer_, bounds);
  if (nullptr == rawptr_) rawptr_ = acc.ptr(bounds.lo);
  return acc;
}

template <typename T, int DIM>
AccessorRW<T, DIM> FutureWrapper::read_write_accessor(const Legion::Rect<DIM>& bounds) const
{
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
  auto acc = AccessorRW<T, DIM>(buffer_, bounds);
  if (nullptr == rawptr_) rawptr_ = acc.ptr(bounds.lo);
  return acc;
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> FutureWrapper::reduce_accessor(int32_t redop_id,
                                                              const Legion::Rect<DIM>& bounds) const
{
  assert(sizeof(typename OP::LHS) == field_size_);
  assert(!read_only_);
  auto acc = AccessorRD<OP, EXCLUSIVE, DIM>(buffer_, bounds);
  if (nullptr == rawptr_) {
    auto& p = bounds.lo;
    rawptr_ = acc.ptr(p);

    if (uninitialized_) {
      auto identity = OP::identity;
      memcpy(rawptr_, &identity, field_size_);
      uninitialized_ = false;
    }
  }
  return acc;
}

template <int32_t DIM>
Legion::Rect<DIM> FutureWrapper::shape() const
{
  return Legion::Rect<DIM>(domain());
}

template <typename VAL>
VAL FutureWrapper::scalar() const
{
  assert(sizeof(VAL) == field_size_);
  if (!read_only_)
    return buffer_.operator Legion::DeferredValue<VAL>().read();
  else
    return future_.get_result<VAL>();
}

template <typename VAL>
void OutputRegionField::return_data(Buffer<VAL>& buffer, size_t num_elements)
{
  assert(!bound_);
  out_.return_data(fid_, buffer, &num_elements);
}

template <typename T, int DIM>
AccessorRO<T, DIM> Store::read_accessor() const
{
  if (is_future_) return future_.read_accessor<T, DIM>();

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_accessor<T, DIM>(transform);
  }
  return region_field_.read_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorWO<T, DIM> Store::write_accessor() const
{
  if (is_future_) return future_.write_accessor<T, DIM>();

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.write_accessor<T, DIM>(transform);
  }
  return region_field_.write_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorRW<T, DIM> Store::read_write_accessor() const
{
  if (is_future_) return future_.read_write_accessor<T, DIM>();

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_write_accessor<T, DIM>(transform);
  }
  return region_field_.read_write_accessor<T, DIM>();
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor() const
{
  if (is_future_) return future_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_);

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> Store::read_accessor(const Legion::Rect<DIM>& bounds) const
{
  if (is_future_) return future_.read_accessor<T, DIM>(bounds);

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> Store::write_accessor(const Legion::Rect<DIM>& bounds) const
{
  if (is_future_) return future_.write_accessor<T, DIM>(bounds);

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.write_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> Store::read_write_accessor(const Legion::Rect<DIM>& bounds) const
{
  if (is_future_) return future_.read_write_accessor<T, DIM>(bounds);

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_write_accessor<T, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor(const Legion::Rect<DIM>& bounds) const
{
  if (is_future_) return future_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds);

  assert(DIM == dim_ || dim_ == 0);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds, transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds);
}

template <int32_t DIM>
Legion::Rect<DIM> Store::shape() const
{
  auto dom = domain();
  if (dom.dim > 0)
    return Legion::Rect<DIM>(dom);
  else {
    auto p = Legion::Point<DIM>::ZEROES();
    return Legion::Rect<DIM>(p, p);
  }
}

template <typename VAL>
VAL Store::scalar() const
{
  assert(is_future_);
  return future_.scalar<VAL>();
}

template <typename VAL>
void Store::return_data(Buffer<VAL>& buffer, size_t num_elements)
{
  assert(is_output_store_);
  output_field_.return_data(buffer, num_elements);
}

}  // namespace legate
