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

#include <memory>

#include "core/utilities/typedefs.h"

namespace legate {

struct Transform {
  virtual Domain transform(const Domain& input) const                           = 0;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const = 0;
  virtual void print(std::ostream& out) const                                   = 0;
};

struct StoreTransform : public Transform {
  virtual ~StoreTransform() {}
  virtual int32_t target_ndim(int32_t source_ndim) const        = 0;
  virtual void find_imaginary_dims(std::vector<int32_t>&) const = 0;
};

struct TransformStack : public Transform {
 public:
  TransformStack() {}
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 std::shared_ptr<TransformStack>&& parent);

 public:
  virtual Domain transform(const Domain& input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  std::unique_ptr<StoreTransform> pop();
  bool identity() const { return nullptr == transform_; }

 public:
  void dump() const;

 public:
  std::vector<int32_t> find_imaginary_dims() const;

 private:
  std::unique_ptr<StoreTransform> transform_{nullptr};
  std::shared_ptr<TransformStack> parent_{nullptr};
};

class Shift : public StoreTransform {
 public:
  Shift(int32_t dim, int64_t offset);

 public:
  virtual Domain transform(const Domain& input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  int64_t offset_;
};

class Promote : public StoreTransform {
 public:
  Promote(int32_t extra_dim, int64_t dim_size);

 public:
  virtual Domain transform(const Domain& input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t extra_dim_;
  int64_t dim_size_;
};

class Project : public StoreTransform {
 public:
  Project(int32_t dim, int64_t coord);
  virtual ~Project() {}

 public:
  virtual Domain transform(const Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  int64_t coord_;
};

class Transpose : public StoreTransform {
 public:
  Transpose(std::vector<int32_t>&& axes);

 public:
  virtual Domain transform(const Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  std::vector<int32_t> axes_;
};

class Delinearize : public StoreTransform {
 public:
  Delinearize(int32_t dim, std::vector<int64_t>&& sizes);

 public:
  virtual Domain transform(const Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t volume_;
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate
