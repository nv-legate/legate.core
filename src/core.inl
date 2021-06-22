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
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor() const
{
  return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, OP::REDOP_ID);
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRO<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  // XXX: there's an interesting fact as to why we need to construct an int32_t value from
  //      OP::REDOP_ID and cannot pass it directly: Since the dispatcher forwards a reference,
  //      passing OP::REDOP_ID directly requires it to be instantiated in the binary so that its
  //      reference can be taken, even when it's a static const member of the class OP.
  //      Since not every class has its static members explicitly instantiated when they are
  //      inline initialized, we should not pass them directly to any function that forwards them.
  return dim_dispatch(transform.transform.m,
                      trans_accesor_fn<ACC, DIM>{},
                      pr_,
                      fid_,
                      int32_t{OP::REDOP_ID},
                      transform);
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::Rect<DIM> &bounds) const
{
  return AccessorRO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::Rect<DIM> &bounds) const
{
  return AccessorWO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(const Legion::Rect<DIM> &bounds) const
{
  return AccessorRW<T, DIM>(pr_, fid_, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(const Legion::Rect<DIM> &bounds) const
{
  return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, OP::REDOP_ID, bounds);
}

template <typename T, int32_t DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Legion::Rect<DIM> &bounds,
                                              const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::Rect<DIM> &bounds,
                                               const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Legion::Rect<DIM> &bounds, const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accesor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  const Legion::Rect<DIM> &bounds, const Legion::DomainAffineTransform &transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  // XXX: there's an interesting fact as to why we need to construct an int32_t value from
  //      OP::REDOP_ID and cannot pass it directly: Since the dispatcher forwards a reference,
  //      passing OP::REDOP_ID directly requires it to be instantiated in the binary so that its
  //      reference can be taken, even when it's a static const member of the class OP.
  //      Since not every class has its static members explicitly instantiated when they are
  //      inline initialized, we should not pass them directly to any function that forwards them.
  return dim_dispatch(transform.transform.m,
                      trans_accesor_fn<ACC, DIM>{},
                      pr_,
                      fid_,
                      int32_t{OP::REDOP_ID},
                      transform,
                      bounds);
}

template <int32_t DIM>
Legion::Rect<DIM> RegionField::shape() const
{
  return Legion::Rect<DIM>(pr_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> Store::read_accessor() const
{
  assert(DIM == dim_);
  if (is_future_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, memkind, sizeof(T), false, false, NULL, sizeof(uint64_t));
  }
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_accessor<T, DIM>(transform);
  }
  return region_field_.read_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorWO<T, DIM> Store::write_accessor() const
{
  assert(DIM == dim_);
  assert(!is_future_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.write_accessor<T, DIM>(transform);
  }
  return region_field_.write_accessor<T, DIM>();
}

template <typename T, int DIM>
AccessorRW<T, DIM> Store::read_write_accessor() const
{
  assert(DIM == dim_);
  assert(!is_future_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_write_accessor<T, DIM>(transform);
  }
  return region_field_.read_write_accessor<T, DIM>();
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor() const
{
  assert(DIM == dim_);
  assert(!is_future_);
  assert(OP::REDOP_ID == redop_id_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>();
}

template <typename T, int DIM>
AccessorRO<T, DIM> Store::read_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(DIM == dim_);
  if (is_future_) {
    auto memkind = Legion::Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(
      future_, bounds, memkind, sizeof(T), false, false, NULL, sizeof(uint64_t));
  }
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> Store::write_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  assert(DIM == dim_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.write_accessor<T, DIM>(bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> Store::read_write_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  assert(DIM == dim_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.read_write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_write_accessor<T, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor(const Legion::Rect<DIM> &bounds) const
{
  assert(!is_future_);
  assert(DIM == dim_);
  assert(OP::REDOP_ID == redop_id_);
  if (nullptr != transform_) {
    auto transform = transform_->inverse_transform(DIM);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(bounds, transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(bounds);
}

template <int32_t DIM>
Legion::Rect<DIM> Store::shape() const
{
  assert(!is_future_);
  auto domain = region_field_.domain();
  if (nullptr != transform_) domain = transform_->transform(domain);
  return Legion::Rect<DIM>(domain);
}

template <typename VAL>
VAL Store::scalar() const
{
  assert(is_future_);
  return future_.get_result<VAL>();
}

}  // namespace legate
