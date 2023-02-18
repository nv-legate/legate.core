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
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"
#include "legion.h"

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

class OutputRegionField {
 public:
  OutputRegionField() {}
  OutputRegionField(const Legion::OutputRegion& out, Legion::FieldID fid);

 public:
  OutputRegionField(OutputRegionField&& other) noexcept;
  OutputRegionField& operator=(OutputRegionField&& other) noexcept;

 private:
  OutputRegionField(const OutputRegionField& other)            = delete;
  OutputRegionField& operator=(const OutputRegionField& other) = delete;

 public:
  bool bound() const { return bound_; }

 public:
  template <typename T, int32_t DIM>
  Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents, bool return_buffer);

 public:
  template <typename T, int32_t DIM>
  void return_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents);
  void make_empty(int32_t dim);

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
                int32_t field_size,
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

 public:
  void initialize_with_identity(int32_t redop_id);

 public:
  ReturnValue pack() const;

 private:
  bool read_only_{true};
  size_t field_size_{0};
  Domain domain_{};
  Legion::Future future_{};
  Legion::UntypedDeferredValue buffer_{};
};

class Store {
 public:
  Store() {}
  Store(int32_t dim,
        int32_t code,
        int32_t redop_id,
        FutureWrapper future,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        int32_t code,
        int32_t redop_id,
        RegionField&& region_field,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        int32_t code,
        OutputRegionField&& output,
        std::shared_ptr<TransformStack>&& transform = nullptr);

 public:
  Store(Store&& other) noexcept;
  Store& operator=(Store&& other) noexcept;

 private:
  Store(const Store& other)            = delete;
  Store& operator=(const Store& other) = delete;

 public:
  bool valid() const;
  bool transformed() const { return !transform_->identity(); }

 public:
  int32_t dim() const { return dim_; }
  template <typename TYPE_CODE = LegateTypeCode>
  TYPE_CODE code() const
  {
    return static_cast<TYPE_CODE>(code_);
  }

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor() const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor() const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor() const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Rect<DIM>& bounds) const;

 public:
  template <typename T, int32_t DIM>
  Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents, bool return_buffer = false);

 public:
  template <int32_t DIM>
  Rect<DIM> shape() const;
  Domain domain() const;

 public:
  bool is_readable() const { return readable_; }
  bool is_writable() const { return writable_; }
  bool is_reducible() const { return reducible_; }

 public:
  template <typename VAL>
  VAL scalar() const;

 public:
  template <typename T, int32_t DIM>
  void return_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents);
  void make_empty();

 public:
  bool is_future() const { return is_future_; }
  bool is_output_store() const { return is_output_store_; }
  ReturnValue pack() const { return future_.pack(); }
  ReturnValue pack_weight() const { return output_field_.pack_weight(); }

 public:
  // TODO: It'd be btter to return a parent store from this method than permanently
  // losing the transform. This requires the backing storages to be referenced by multiple
  // stores, which isn't possible as they use move-only types.
  void remove_transform();

 private:
  void check_valid_return() const;
  void check_buffer_dimension(const int32_t dim) const;
  void check_accessor_dimension(const int32_t dim) const;

 private:
  bool is_future_{false};
  bool is_output_store_{false};
  int32_t dim_{-1};
  int32_t code_{-1};
  int32_t redop_id_{-1};

 private:
  FutureWrapper future_;
  RegionField region_field_;
  OutputRegionField output_field_;

 private:
  std::shared_ptr<TransformStack> transform_{nullptr};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

}  // namespace legate

#include "core/data/store.inl"
