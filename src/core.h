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

#pragma once

#include <memory>

#include "legate.h"

namespace legate {

class UntypedPoint {
 public:
  UntypedPoint() noexcept {}
  ~UntypedPoint();

 public:
  template <int32_t N>
  UntypedPoint(const Legion::Point<N> &point) : N_(N), point_(new Legion::Point<N>(point))
  {
  }

 public:
  UntypedPoint(const UntypedPoint &other);
  UntypedPoint &operator=(const UntypedPoint &other);

 public:
  UntypedPoint(UntypedPoint &&other) noexcept;
  UntypedPoint &operator=(UntypedPoint &&other) noexcept;

 public:
  int32_t dim() const noexcept { return N_; }
  bool exists() const noexcept { return nullptr != point_; }

 public:
  template <int32_t N>
  Legion::Point<N> to_point() const
  {
    assert(N_ == N);
    return *static_cast<Legion::Point<N> *>(point_);
  }

 private:
  struct copy_fn {
    template <int32_t N>
    void *operator()(void *point)
    {
      return new Legion::Point<N>(*static_cast<Legion::Point<N> *>(point));
    }
  };
  void copy(const UntypedPoint &other);
  void move(UntypedPoint &&other);
  struct destroy_fn {
    template <int32_t N>
    void operator()(void *point)
    {
      delete static_cast<Legion::Point<N> *>(point);
    }
  };
  void destroy();

 private:
  int32_t N_{-1};
  void *point_{nullptr};
};

std::ostream &operator<<(std::ostream &os, const UntypedPoint &point);

class Shape {
 public:
  Shape() noexcept {}
  ~Shape();

 public:
  template <int32_t N>
  Shape(const Legion::Rect<N> &rect) : N_(N), rect_(new Legion::Rect<N>(rect))
  {
  }

 public:
  Shape(const Shape &other);
  Shape &operator=(const Shape &other);

 public:
  Shape(Shape &&other) noexcept;
  Shape &operator=(Shape &&other) noexcept;

 public:
  int32_t dim() const noexcept { return N_; }
  bool exists() const noexcept { return nullptr != rect_; }

 public:
  template <int32_t N>
  Legion::Rect<N> to_rect() const
  {
    assert(N_ == N);
    return *static_cast<Legion::Rect<N> *>(rect_);
  }

 private:
  struct copy_fn {
    template <int32_t N>
    void *operator()(void *rect)
    {
      return new Legion::Rect<N>(*static_cast<Legion::Rect<N> *>(rect));
    }
  };
  void copy(const Shape &other);
  void move(Shape &&other);
  struct destroy_fn {
    template <int32_t N>
    void operator()(void *rect)
    {
      delete static_cast<Legion::Rect<N> *>(rect);
    }
  };
  void destroy();

 private:
  int32_t N_{-1};
  void *rect_{nullptr};
};

std::ostream &operator<<(std::ostream &os, const Shape &shape);

class Transform {
 public:
  Transform() {}
  Transform(std::unique_ptr<Transform> &&parent);
  ~Transform() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &input) const           = 0;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const = 0;

 protected:
  std::unique_ptr<Transform> parent_{nullptr};
};

class Shift : public Transform {
 public:
  Shift(int32_t dim, int64_t offset, std::unique_ptr<Transform> &&parent = nullptr);
  virtual ~Shift() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;

 private:
  int32_t dim_;
  int64_t offset_;
};

class Promote : public Transform {
 public:
  Promote(int32_t extra_dim, int64_t dim_size, std::unique_ptr<Transform> &&parent = nullptr);
  virtual ~Promote() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;

 private:
  int32_t extra_dim_;
  int64_t dim_size_;
};

class Project : public Transform {
 public:
  Project(int32_t dim, int64_t coord, std::unique_ptr<Transform> &&parent = nullptr);
  virtual ~Project() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;

 private:
  int32_t dim_;
  int64_t coord_;
};

class Transpose : public Transform {
 public:
  Transpose(std::vector<int32_t> &&axes, std::unique_ptr<Transform> &&parent = nullptr);
  virtual ~Transpose() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;

 private:
  std::vector<int32_t> axes_;
};

class Delinearize : public Transform {
 public:
  Delinearize(int32_t dim,
              std::vector<int64_t> &&sizes,
              std::unique_ptr<Transform> &&parent = nullptr);
  virtual ~Delinearize() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain &domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;

 private:
  int32_t dim_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t volume_;
};

class RegionField {
 public:
  RegionField() {}
  RegionField(int32_t dim, const Legion::PhysicalRegion &pr, Legion::FieldID fid);

 public:
  RegionField(RegionField &&other) noexcept;
  RegionField &operator=(RegionField &&other) noexcept;

 private:
  RegionField(const RegionField &other) = delete;
  RegionField &operator=(const RegionField &other) = delete;

 public:
  int32_t dim() const { return dim_; }

 private:
  template <typename ACC, int32_t N>
  struct trans_accesor_fn {
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion &pr,
                   Legion::FieldID fid,
                   const Legion::AffineTransform<M, N> &transform)
    {
      return ACC(pr, fid, transform);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion &pr,
                   Legion::FieldID fid,
                   const Legion::AffineTransform<M, N> &transform,
                   const Legion::Rect<N> &bounds)
    {
      return ACC(pr, fid, transform, bounds);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion &pr,
                   Legion::FieldID fid,
                   int32_t redop_id,
                   const Legion::AffineTransform<M, N> &transform)
    {
      return ACC(pr, fid, redop_id, transform);
    }
    template <int32_t M>
    ACC operator()(const Legion::PhysicalRegion &pr,
                   Legion::FieldID fid,
                   int32_t redop_id,
                   const Legion::AffineTransform<M, N> &transform,
                   const Legion::Rect<N> &bounds)
    {
      return ACC(pr, fid, redop_id, transform, bounds);
    }
  };

  struct get_domain_fn {
    template <int32_t DIM>
    Legion::Domain operator()(const Legion::PhysicalRegion &pr)
    {
      return Legion::Domain(pr.get_bounds<DIM, Legion::coord_t>());
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
  AccessorRO<T, DIM> read_accessor(const Legion::DomainAffineTransform &transform) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::DomainAffineTransform &transform) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::DomainAffineTransform &transform) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(
    int32_t redop_id, const Legion::DomainAffineTransform &transform) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(int32_t redop_id,
                                                 const Legion::Rect<DIM> &bounds) const;

 public:
  template <typename T, int32_t DIM>
  AccessorRO<T, DIM> read_accessor(const Legion::Rect<DIM> &bounds,
                                   const Legion::DomainAffineTransform &transform) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::Rect<DIM> &bounds,
                                    const Legion::DomainAffineTransform &transform) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::Rect<DIM> &bounds,
                                         const Legion::DomainAffineTransform &transform) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(
    int32_t redop_id,
    const Legion::Rect<DIM> &bounds,
    const Legion::DomainAffineTransform &transform) const;

 public:
  template <int32_t DIM>
  Legion::Rect<DIM> shape() const;
  Legion::Domain domain() const;

 private:
  int32_t dim_{-1};
  Legion::PhysicalRegion pr_{};
  Legion::FieldID fid_{-1U};
};

class Store {
 public:
  Store() {}
  Store(int32_t dim,
        LegateTypeCode code,
        int32_t redop_id,
        Legion::Future future,
        std::unique_ptr<Transform> transform = nullptr);
  Store(int32_t dim,
        LegateTypeCode code,
        int32_t redop_id,
        RegionField &&region_field,
        std::unique_ptr<Transform> transform = nullptr);

 public:
  Store(Store &&other) noexcept;
  Store &operator=(Store &&other) noexcept;

 private:
  Store(const Store &other) = delete;
  Store &operator=(const Store &other) = delete;

 public:
  int32_t dim() const { return dim_; }
  LegateTypeCode code() const { return code_; }

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
  AccessorRO<T, DIM> read_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorWO<T, DIM> write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> read_write_accessor(const Legion::Rect<DIM> &bounds) const;
  template <typename OP, bool EXCLUSIVE, int32_t DIM>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Legion::Rect<DIM> &bounds) const;

 public:
  template <int32_t DIM>
  Legion::Rect<DIM> shape() const;
  Legion::Domain domain() const;

 public:
  template <typename VAL>
  VAL scalar() const;

 private:
  bool is_future_{false};
  int32_t dim_{-1};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  int32_t redop_id_{-1};
  Legion::Future future_;
  RegionField region_field_;
  std::unique_ptr<Transform> transform_{nullptr};
};

}  // namespace legate

#include "core.inl"
