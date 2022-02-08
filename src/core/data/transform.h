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

#include "legion.h"

namespace legate {

class StoreTransform {
 public:
  StoreTransform() {}
  StoreTransform(std::shared_ptr<StoreTransform> parent);
  virtual ~StoreTransform() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& input) const           = 0;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const = 0;
  virtual void print(std::ostream& out) const                                   = 0;

 protected:
  std::shared_ptr<StoreTransform> parent_{nullptr};
};

std::ostream& operator<<(std::ostream& out, const StoreTransform& transform);

class Shift : public StoreTransform {
 public:
  Shift(int32_t dim, int64_t offset, std::shared_ptr<StoreTransform> parent = nullptr);
  virtual ~Shift() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 private:
  int32_t dim_;
  int64_t offset_;
};

class Promote : public StoreTransform {
 public:
  Promote(int32_t extra_dim, int64_t dim_size, std::shared_ptr<StoreTransform> parent = nullptr);
  virtual ~Promote() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& input) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 private:
  int32_t extra_dim_;
  int64_t dim_size_;
};

class Project : public StoreTransform {
 public:
  Project(int32_t dim, int64_t coord, std::shared_ptr<StoreTransform> parent = nullptr);
  virtual ~Project() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 private:
  int32_t dim_;
  int64_t coord_;
};

class Transpose : public StoreTransform {
 public:
  Transpose(std::vector<int32_t>&& axes, std::shared_ptr<StoreTransform> parent = nullptr);
  virtual ~Transpose() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 private:
  std::vector<int32_t> axes_;
};

class Delinearize : public StoreTransform {
 public:
  Delinearize(int32_t dim,
              std::vector<int64_t>&& sizes,
              std::shared_ptr<StoreTransform> parent = nullptr);
  virtual ~Delinearize() {}

 public:
  virtual Legion::Domain transform(const Legion::Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual void print(std::ostream& out) const override;

 private:
  int32_t dim_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t volume_;
};

}  // namespace legate
