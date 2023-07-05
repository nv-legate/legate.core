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
  return dim_dispatch(transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  int32_t redop_id, const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, redop_id, transform);
}

template <typename T, int DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Rect<DIM>& bounds) const
{
  return AccessorRO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Rect<DIM>& bounds) const
{
  return AccessorWO<T, DIM>(pr_, fid_, bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(const Rect<DIM>& bounds) const
{
  return AccessorRW<T, DIM>(pr_, fid_, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(int32_t redop_id,
                                                            const Rect<DIM>& bounds) const
{
  return AccessorRD<OP, EXCLUSIVE, DIM>(pr_, fid_, redop_id, bounds);
}

template <typename T, int32_t DIM>
AccessorRO<T, DIM> RegionField::read_accessor(const Rect<DIM>& bounds,
                                              const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorWO<T, DIM> RegionField::write_accessor(const Rect<DIM>& bounds,
                                               const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorWO<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename T, int32_t DIM>
AccessorRW<T, DIM> RegionField::read_write_accessor(
  const Rect<DIM>& bounds, const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRW<T, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, transform, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> RegionField::reduce_accessor(
  int32_t redop_id, const Rect<DIM>& bounds, const Legion::DomainAffineTransform& transform) const
{
  using ACC = AccessorRD<OP, EXCLUSIVE, DIM>;
  return dim_dispatch(
    transform.transform.m, trans_accessor_fn<ACC, DIM>{}, pr_, fid_, redop_id, transform, bounds);
}

template <int32_t DIM>
Rect<DIM> RegionField::shape() const
{
  return Rect<DIM>(pr_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> FutureWrapper::read_accessor() const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
#endif
  if (read_only_) {
    auto memkind = Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, memkind);
  } else
    return AccessorRO<T, DIM>(buffer_);
}

template <typename T, int DIM>
AccessorWO<T, DIM> FutureWrapper::write_accessor() const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
#endif
  return AccessorWO<T, DIM>(buffer_);
}

template <typename T, int DIM>
AccessorRW<T, DIM> FutureWrapper::read_write_accessor() const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
#endif
  return AccessorRW<T, DIM>(buffer_);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> FutureWrapper::reduce_accessor(int32_t redop_id) const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(typename OP::LHS) == field_size_);
  assert(!read_only_);
#endif
  return AccessorRD<OP, EXCLUSIVE, DIM>(buffer_);
}

template <typename T, int DIM>
AccessorRO<T, DIM> FutureWrapper::read_accessor(const Rect<DIM>& bounds) const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
#endif
  if (read_only_) {
    auto memkind = Memory::Kind::NO_MEMKIND;
    return AccessorRO<T, DIM>(future_, bounds, memkind);
  } else
    return AccessorRO<T, DIM>(buffer_, bounds);
}

template <typename T, int DIM>
AccessorWO<T, DIM> FutureWrapper::write_accessor(const Rect<DIM>& bounds) const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
#endif
  return AccessorWO<T, DIM>(buffer_, bounds);
}

template <typename T, int DIM>
AccessorRW<T, DIM> FutureWrapper::read_write_accessor(const Rect<DIM>& bounds) const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(T) == field_size_);
  assert(!read_only_);
#endif
  return AccessorRW<T, DIM>(buffer_, bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM>
AccessorRD<OP, EXCLUSIVE, DIM> FutureWrapper::reduce_accessor(int32_t redop_id,
                                                              const Rect<DIM>& bounds) const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(typename OP::LHS) == field_size_);
  assert(!read_only_);
#endif
  return AccessorRD<OP, EXCLUSIVE, DIM>(buffer_, bounds);
}

template <int32_t DIM>
Rect<DIM> FutureWrapper::shape() const
{
  return Rect<DIM>(domain());
}

template <typename VAL>
VAL FutureWrapper::scalar() const
{
#ifdef DEBUG_LEGATE
  assert(sizeof(VAL) == field_size_);
#endif
  if (!read_only_)
    return buffer_.operator Legion::DeferredValue<VAL>().read();
  else
    return future_.get_result<VAL>();
}

template <typename T, int32_t DIM>
Buffer<T, DIM> UnboundRegionField::create_output_buffer(const Point<DIM>& extents, bool bind_buffer)
{
  if (bind_buffer) {
#ifdef DEBUG_LEGATE
    assert(!bound_);
#endif
    // We will use this value only when the unbound store is 1D
    update_num_elements(extents[0]);
    bound_ = true;
  }
  return out_.create_buffer<T, DIM>(extents, fid_, nullptr, bind_buffer);
}

template <typename T, int32_t DIM>
void UnboundRegionField::bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents)
{
#ifdef DEBUG_LEGATE
  assert(!bound_);
#endif
  out_.return_data(extents, fid_, buffer);
  // We will use this value only when the unbound store is 1D
  update_num_elements(extents[0]);
  bound_ = true;
}

template <typename T>
void Store::check_accessor_type() const
{
  auto in_type = legate_type_code_of<T>;
  if (in_type == this->code()) return;
  // Test exact match for primitive types
  if (in_type != Type::Code::INVALID) {
    throw std::invalid_argument(
      "Type mismatch: " + PrimitiveType(in_type).to_string() + " accessor to a " +
      type().to_string() +
      " store. Disable type checking via accessor template parameter if this is intended.");
  }
  // Test size matches for other types
  if (sizeof(T) != type().size()) {
    throw std::invalid_argument(
      "Type size mismatch: store type " + type().to_string() + " has size " +
      std::to_string(type().size()) + ", requested type has size " + std::to_string(sizeof(T)) +
      ". Disable type checking via accessor template parameter if this is intended.");
  }
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> Store::read_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.read_accessor<T, DIM>(shape<DIM>());

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_accessor<T, DIM>(shape<DIM>(), transform);
  }
  return region_field_.read_accessor<T, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> Store::write_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.write_accessor<T, DIM>(shape<DIM>());

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.write_accessor<T, DIM>(shape<DIM>(), transform);
  }
  return region_field_.write_accessor<T, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> Store::read_write_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.read_write_accessor<T, DIM>(shape<DIM>());

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_write_accessor<T, DIM>(shape<DIM>(), transform);
  }
  return region_field_.read_write_accessor<T, DIM>(shape<DIM>());
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor() const
{
  using T = typename OP::LHS;
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, shape<DIM>());

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, shape<DIM>(), transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> Store::read_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.read_accessor<T, DIM>(bounds);

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_accessor<T, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> Store::write_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.write_accessor<T, DIM>(bounds);

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.write_accessor<T, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> Store::read_write_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.read_write_accessor<T, DIM>(bounds);

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.read_write_accessor<T, DIM>(bounds, transform);
  }
  return region_field_.read_write_accessor<T, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor(const Rect<DIM>& bounds) const
{
  using T = typename OP::LHS;
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future_) return future_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds);

  if (!transform_->identity()) {
    auto transform = transform_->inverse_transform(dim_);
    return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds, transform);
  }
  return region_field_.reduce_accessor<OP, EXCLUSIVE, DIM>(redop_id_, bounds);
}

template <typename T, int32_t DIM>
Buffer<T, DIM> Store::create_output_buffer(const Point<DIM>& extents, bool bind_buffer /*= false*/)
{
  check_valid_binding();
  check_buffer_dimension(DIM);
  return unbound_field_.create_output_buffer<T, DIM>(extents, bind_buffer);
}

template <int32_t DIM>
Rect<DIM> Store::shape() const
{
  check_shape_dimension(DIM);
  if (dim_ > 0) {
    return domain().bounds<DIM, Legion::coord_t>();
  } else {
    auto p = Point<DIM>::ZEROES();
    return Rect<DIM>(p, p);
  }
}

template <typename VAL>
VAL Store::scalar() const
{
  if (!is_future_) throw std::invalid_argument("Scalars can be retrieved only from scalar stores");
  return future_.scalar<VAL>();
}

template <typename T, int32_t DIM>
void Store::bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents)
{
  check_valid_binding();
  check_buffer_dimension(DIM);
  unbound_field_.bind_data(buffer, extents);
}

}  // namespace legate
