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
#include <tuple>

#include "core/data/scalar.h"
#include "core/data/transform.h"
#include "core/runtime/context.h"

namespace legate {
namespace mapping {

class RegionField {
 public:
  using Id = std::tuple<bool, uint32_t, Legion::FieldID>;

 public:
  RegionField() {}
  RegionField(const Legion::RegionRequirement* req, int32_t dim, uint32_t idx, Legion::FieldID fid);

 public:
  RegionField(const RegionField& other)            = default;
  RegionField& operator=(const RegionField& other) = default;

 public:
  bool can_colocate_with(const RegionField& other) const;

 public:
  template <int32_t DIM>
  Rect<DIM> shape(Legion::Mapping::MapperRuntime* runtime,
                  const Legion::Mapping::MapperContext context) const;

 public:
  Domain domain(Legion::Mapping::MapperRuntime* runtime,
                const Legion::Mapping::MapperContext context) const;

 public:
  bool operator==(const RegionField& other) const;

 public:
  Id unique_id() const { return std::make_tuple(unbound(), idx_, fid_); }

 public:
  int32_t dim() const { return dim_; }
  uint32_t index() const { return idx_; }
  Legion::FieldID field_id() const { return fid_; }
  bool unbound() const { return dim_ < 0; }

 public:
  const Legion::RegionRequirement* get_requirement() const { return req_; }
  Legion::IndexSpace get_index_space() const;

 private:
  const Legion::RegionRequirement* req_{nullptr};
  int32_t dim_{-1};
  uint32_t idx_{-1U};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() {}
  FutureWrapper(uint32_t idx, const Domain& domain);

 public:
  FutureWrapper(const FutureWrapper& other)            = default;
  FutureWrapper& operator=(const FutureWrapper& other) = default;

 public:
  int32_t dim() const { return domain_.dim; }
  uint32_t index() const { return idx_; }

 public:
  template <int32_t DIM>
  Rect<DIM> shape() const;
  Domain domain() const;

 private:
  uint32_t idx_{-1U};
  Domain domain_{};
};

class Store {
 public:
  Store() {}
  Store(int32_t dim,
        LegateTypeCode code,
        FutureWrapper future,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  Store(Legion::Mapping::MapperRuntime* runtime,
        const Legion::Mapping::MapperContext context,
        int32_t dim,
        LegateTypeCode code,
        int32_t redop_id,
        const RegionField& region_field,
        bool is_output_store                        = false,
        std::shared_ptr<TransformStack>&& transform = nullptr);
  // A special constructor to create a mapper view of a store from a region requirement
  Store(Legion::Mapping::MapperRuntime* runtime,
        const Legion::Mapping::MapperContext context,
        const Legion::RegionRequirement* requirement);

 public:
  Store(const Store& other)            = default;
  Store& operator=(const Store& other) = default;

 public:
  Store(Store&& other)            = default;
  Store& operator=(Store&& other) = default;

 public:
  bool is_future() const { return is_future_; }
  bool unbound() const { return is_output_store_; }
  int32_t dim() const { return dim_; }

 public:
  bool is_reduction() const { return redop_id_ > 0; }
  int32_t redop() const { return redop_id_; }

 public:
  bool can_colocate_with(const Store& other) const;
  const RegionField& region_field() const;
  const FutureWrapper& future() const;

 public:
  RegionField::Id unique_region_field_id() const;
  uint32_t requirement_index() const;
  uint32_t future_index() const;

 public:
  template <int32_t DIM>
  Rect<DIM> shape() const;

 public:
  Domain domain() const;

 private:
  bool is_future_{false};
  bool is_output_store_{false};
  int32_t dim_{-1};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  int32_t redop_id_{-1};

 private:
  FutureWrapper future_;
  RegionField region_field_;

 private:
  std::shared_ptr<TransformStack> transform_{nullptr};

 private:
  Legion::Mapping::MapperRuntime* runtime_{nullptr};
  Legion::Mapping::MapperContext context_{nullptr};
};

class Task {
 public:
  Task(const Legion::Task* task,
       const LibraryContext& library,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  int64_t task_id() const;

 public:
  const std::vector<Store>& inputs() const { return inputs_; }
  const std::vector<Store>& outputs() const { return outputs_; }
  const std::vector<Store>& reductions() const { return reductions_; }
  const std::vector<Scalar>& scalars() const { return scalars_; }

 public:
  DomainPoint point() const { return task_->index_point; }

 private:
  const LibraryContext& library_;
  const Legion::Task* task_;

 private:
  std::vector<Store> inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
};

class Copy {
 public:
  Copy(const Legion::Copy* copy,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  const std::vector<Store>& inputs() const { return inputs_; }
  const std::vector<Store>& outputs() const { return outputs_; }
  const std::vector<Store>& input_indirections() const { return input_indirections_; }
  const std::vector<Store>& output_indirections() const { return output_indirections_; }

 public:
  DomainPoint point() const { return copy_->index_point; }

 private:
  const Legion::Copy* copy_;

 private:
  std::vector<Store> inputs_;
  std::vector<Store> outputs_;
  std::vector<Store> input_indirections_;
  std::vector<Store> output_indirections_;
};

}  // namespace mapping
}  // namespace legate

#include "core/mapping/operation.inl"
