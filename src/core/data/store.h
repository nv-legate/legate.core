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

#pragma once

#include "core/data/buffer.h"
#include "core/data/transform.h"
#include "core/task/return.h"
#include "core/type/type_info.h"
#include "core/type/type_traits.h"
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"
#include "legion.h"

/** @defgroup data Data abstractions and allocators
 */

/**
 * @file
 * @brief Class definition for legate::Store
 */

namespace legate {

class RegionField {
 public:
  RegionField() {}
  RegionField(int32_t dim, const Legion::PhysicalRegion& pr, Legion::FieldID fid);

 public:
  RegionField(RegionField&& other) noexcept;
  RegionField& operator=(RegionField&& other) noexcept;

 private:
  RegionField(const RegionField& other)            = delete;
  RegionField& operator=(const RegionField& other) = delete;

 public:
  bool valid() const;

 public:
  int32_t dim() const { return dim_; }

 private:
  template <typename ACC, int32_t N>
  struct trans_accessor_fn {
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion& pr,
                   Legion::FieldID fid,
                   const Legion::AffineTransform<M, N>& transform)
    {
      return ACC(pr, fid, transform);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion& pr,
                   Legion::FieldID fid,
                   const Legion::AffineTransform<M, N>& transform,
                   const Rect<N>& bounds)
    {
      return ACC(pr, fid, transform, bounds);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion& pr,
                   Legion::FieldID fid,
                   int32_t redop_id,
                   const Legion::AffineTransform<M, N>& transform)
    {
      return ACC(pr, fid, redop_id, transform);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion& pr,
                   Legion::FieldID fid,
                   int32_t redop_id,
                   const Legion::AffineTransform<M, N>& transform,
                   const Rect<N>& bounds)
    {
      return ACC(pr, fid, redop_id, transform, bounds);
    }
  };

  struct get_domain_fn {
    template <int32_t DIM>
    Domain operator()(const Legion::PhysicalRegion& pr)
    {
      return Domain(pr.get_bounds<DIM, Legion::coord_t>());
    }
  };

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor() const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor() const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor() const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(int32_t redop_id) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Legion::DomainAffineTransform& transform) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::DomainAffineTransform& transform) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::DomainAffineTransform& transform) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(
    int32_t redop_id, const Legion::DomainAffineTransform& transform) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(int32_t redop_id, const Rect<DIM>& bounds) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds,
                                   const Legion::DomainAffineTransform& transform) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds,
                                    const Legion::DomainAffineTransform& transform) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds,
                                         const Legion::DomainAffineTransform& transform) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(
    int32_t redop_id,
    const Rect<DIM>& bounds,
    const Legion::DomainAffineTransform& transform) const;

 public:
  template <int32_t DIM>
  Rect<DIM> shape() const;
  Domain domain() const;

 public:
  bool is_readable() const { return readable_; }
  bool is_writable() const { return writable_; }
  bool is_reducible() const { return reducible_; }

 private:
  int32_t dim_{-1};
  Legion::PhysicalRegion pr_{};
  Legion::FieldID fid_{-1U};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

class UnboundRegionField {
 public:
  UnboundRegionField() {}
  UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid);

 public:
  UnboundRegionField(UnboundRegionField&& other) noexcept;
  UnboundRegionField& operator=(UnboundRegionField&& other) noexcept;

 private:
  UnboundRegionField(const UnboundRegionField& other)            = delete;
  UnboundRegionField& operator=(const UnboundRegionField& other) = delete;

 public:
  bool bound() const { return bound_; }

 public:
  template <typename T, int32_t DIM>
  Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents, bool bind_buffer);

 public:
  template <typename T, int32_t DIM>
  void bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents);
  void bind_empty_data(int32_t dim);

 public:
  ReturnValue pack_weight() const;

 private:
  void update_num_elements(size_t num_elements);

 private:
  bool bound_{false};
  Legion::UntypedDeferredValue num_elements_;
  Legion::OutputRegion out_{};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() {}
  FutureWrapper(bool read_only,
                uint32_t field_size,
                Domain domain,
                Legion::Future future,
                bool initialize = false);

 public:
  FutureWrapper(const FutureWrapper& other) noexcept;
  FutureWrapper& operator=(const FutureWrapper& other) noexcept;

 public:
  int32_t dim() const { return domain_.dim; }

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor() const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor() const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor() const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(int32_t redop_id) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(int32_t redop_id, const Rect<DIM>& bounds) const;

 public:
  template <typename VAL>
  VAL scalar() const;

 public:
  template <int32_t DIM>
  Rect<DIM> shape() const;
  Domain domain() const;
  bool valid() const { return future_.valid(); }

 public:
  void initialize_with_identity(int32_t redop_id);

 public:
  ReturnValue pack() const;

 private:
  bool read_only_{true};
  uint32_t field_size_{0};
  Domain domain_{};
  Legion::Future future_{};
  Legion::UntypedDeferredValue buffer_{};
};

/**
 * @ingroup data
 *
 * @brief A multi-dimensional data container storing task data
 */
class Store {
 public:
  Store() {}
  Store(int32_t dim,
        std::unique_ptr<Type> type,
        int32_t redop_id,
        FutureWrapper future,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        std::unique_ptr<Type> type,
        int32_t redop_id,
        RegionField&& region_field,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        std::unique_ptr<Type> type,
        UnboundRegionField&& unbound_field,
        std::shared_ptr<TransformStack>&& transform = nullptr);

 public:
  Store(Store&& other) noexcept;
  Store& operator=(Store&& other) noexcept;

 private:
  Store(const Store& other)            = delete;
  Store& operator=(const Store& other) = delete;

 public:
  /**
   * @brief Indicates whether the store is valid. A store passed to a task can be invalid
   * only for reducer tasks for tree reduction.
   *
   * @return true The store is valid
   * @return false The store is invalid and cannot be used in any data access
   */
  bool valid() const;
  /**
   * @brief Indicates whether the store is transformed in any way.
   *
   * @return true The store is transformed
   * @return false The store is not transformed
   */
  bool transformed() const { return !transform_->identity(); }

 public:
  /**
   * @brief Returns the dimension of the store
   *
   * @return The store's dimension
   */
  int32_t dim() const { return dim_; }
  /**
   * @brief Returns the type metadata of the store
   *
   * @return The store's type metadata
   */
  const Type& type() const { return *type_; }
  /**
   * @brief Returns the type code of the store
   *
   * @return The store's type code
   */
  template <typename TYPE_CODE = Type::Code>
  TYPE_CODE code() const
  {
    return static_cast<TYPE_CODE>(type_->code);
  }

 public:
  /**
   * @brief Returns a read-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A read-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRO<T, DIM> read_accessor() const;
  /**
   * @brief Returns a write-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A write-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorWO<T, DIM> write_accessor() const;
  /**
   * @brief Returns a read-write accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A read-write accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRW<T, DIM> read_write_accessor() const;
  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @tparam OP Reduction operator class. For details about reduction operators, See
   * LibraryContext::register_reduction_operator.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A reduction accessor to the store
   */
  template <typename OP, bool EXCLUSIVE, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

 public:
  /**
   * @brief Returns a read-only accessor to the store for specific bounds.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A read-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a write-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A write-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a read-write accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A read-write accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @tparam OP Reduction operator class. For details about reduction operators, See
   * LibraryContext::register_reduction_operator.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A reduction accessor to the store
   */
  template <typename OP, bool EXCLUSIVE, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Rect<DIM>& bounds) const;

 public:
  /**
   * @brief Creates a buffer of specified extents for the unbound store. The returned
   * buffer is always consistent with the mapping policy for the store. Can be invoked
   * multiple times unless `bind_buffer` is true.
   *
   * @param extents Extents of the buffer
   *
   * @param bind_buffer If the value is true, the created buffer will be bound
   * to the store upon return
   *
   * @return A reduction accessor to the store
   */
  template <typename T, int32_t DIM>
  Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents, bool bind_buffer = false);

 public:
  /**
   * @brief Returns the store's domain
   *
   * @return Store's domain
   */
  template <int32_t DIM>
  Rect<DIM> shape() const;
  /**
   * @brief Returns the store's domain in a dimension-erased domain type
   *
   * @return Store's domain in a dimension-erased domain type
   */
  Domain domain() const;

 public:
  /**
   * @brief Indicates whether the store can have a read accessor
   *
   * @return true The store can have a read accessor
   * @return false The store cannot have a read accesor
   */
  bool is_readable() const { return readable_; }
  /**
   * @brief Indicates whether the store can have a write accessor
   *
   * @return true The store can have a write accessor
   * @return false The store cannot have a write accesor
   */
  bool is_writable() const { return writable_; }
  /**
   * @brief Indicates whether the store can have a reduction accessor
   *
   * @return true The store can have a reduction accessor
   * @return false The store cannot have a reduction accesor
   */
  bool is_reducible() const { return reducible_; }

 public:
  /**
   * @brief Returns the scalar value stored in the store.
   *
   * The requested type must match with the store's data type. If the store is not
   * backed by the future, the runtime will fail with an error message.
   *
   * @tparam VAL Type of the scalar value
   *
   * @return The scalar value stored in the store
   */
  template <typename VAL>
  VAL scalar() const;

 public:
  /**
   * @brief Binds a buffer to the store. Valid only when the store is unbound and
   * has not yet been bound to another buffer. The buffer must be consistent with
   * the mapping policy for the store. Recommend that the buffer be created by
   * a `create_output_buffer` call.
   *
   * @param buffer Buffer to bind to the store
   *
   * @param extents Extents of the buffer. Passing extents smaller than the actual
   * extents of the buffer is legal; the runtime uses the passed extents as the
   * extents of this store.
   *
   */
  template <typename T, int32_t DIM>
  void bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents);
  /**
   * @brief Makes the unbound store empty. Valid only when the store is unbound and
   * has not yet been bound to another buffer.
   */
  void bind_empty_data();

 public:
  /**
   * @brief Indicates whether the store is backed by a future
   * (i.e., a container for scalar value)
   *
   * @return true The store is backed by a future
   * @return false The store is backed by a region field
   */
  bool is_future() const { return is_future_; }
  /**
   * @brief Indicates whether the store is an unbound store. The value DOES NOT indicate
   * that the store has already assigned to a buffer; i.e., the store may have been assigned
   * to a buffer even when this function returns `true`.
   *
   * @return true The store is an unbound store
   * @return false The store is a normal store
   */
  bool is_unbound_store() const { return is_unbound_store_; }
  ReturnValue pack() const { return future_.pack(); }
  ReturnValue pack_weight() const { return unbound_field_.pack_weight(); }

 public:
  // TODO: It'd be btter to return a parent store from this method than permanently
  // losing the transform. This requires the backing storages to be referenced by multiple
  // stores, which isn't possible as they use move-only types.
  void remove_transform();

 private:
  void check_accessor_dimension(const int32_t dim) const;
  void check_buffer_dimension(const int32_t dim) const;
  void check_shape_dimension(const int32_t dim) const;
  void check_valid_binding() const;
  template <typename T>
  void check_accessor_type() const;

 private:
  bool is_future_{false};
  bool is_unbound_store_{false};
  int32_t dim_{-1};
  std::unique_ptr<Type> type_{nullptr};
  int32_t redop_id_{-1};

 private:
  FutureWrapper future_;
  RegionField region_field_;
  UnboundRegionField unbound_field_;

 private:
  std::shared_ptr<TransformStack> transform_{nullptr};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

}  // namespace legate

#include "core/data/store.inl"
