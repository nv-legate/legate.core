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

#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "dispatch.h"
#include "legate.h"
#include "projection.h"

using namespace Legion;

namespace legate {

uint32_t get_reduction_functor_id(const int32_t src_dim, const int32_t tgt_dim, const uint32_t mask)
{
  assert(src_dim >= tgt_dim);
  return (mask << 8) | (src_dim << 4) | (tgt_dim);
}

uint32_t get_transpose_functor_id(const int32_t dim, const uint32_t id)
{
  return LEGATE_CORE_FIRST_TRANSPOSE_FUNCTOR | id << 4 | dim;
}

static std::unordered_map<ProjectionID, LegateProjectionFunctor*> functors;

LegateProjectionFunctor::LegateProjectionFunctor(Runtime* rt) : ProjectionFunctor(rt) {}

LogicalRegion LegateProjectionFunctor::project(LogicalPartition upper_bound,
                                               const DomainPoint& point,
                                               const Domain& launch_domain)
{
  const DomainPoint dp = project_point(point, launch_domain);
  if (runtime->has_logical_subregion_by_color(upper_bound, dp))
    return runtime->get_logical_subregion_by_color(upper_bound, dp);
  else
    return LogicalRegion::NO_REGION;
}

// This family of functors project points in a higher dimension space to those in a lower one,
// or drop some coordinates out of the points within the same space.
template <int32_t SRC_DIM, int32_t TGT_DIM>
class ReductionFunctor : public LegateProjectionFunctor {
 public:
  ReductionFunctor(Runtime* runtime, uint32_t mask);

 public:
  virtual DomainPoint project_point(const DomainPoint& point,
                                    const Domain& launch_domain) const override
  {
    return DomainPoint(transform_ * Point<SRC_DIM>(point));
  }

 public:
  static Transform<TGT_DIM, SRC_DIM> create_transform(uint32_t mask);

 private:
  const Transform<TGT_DIM, SRC_DIM> transform_;
};

template <int32_t SRC_DIM, int32_t TGT_DIM>
ReductionFunctor<SRC_DIM, TGT_DIM>::ReductionFunctor(Runtime* runtime, uint32_t mask)
  : LegateProjectionFunctor(runtime), transform_(create_transform(mask))
{
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
/*static*/ Transform<TGT_DIM, SRC_DIM> ReductionFunctor<SRC_DIM, TGT_DIM>::create_transform(
  uint32_t mask)
{
  Transform<TGT_DIM, SRC_DIM> transform;

  for (int32_t src_dim = 0; src_dim < SRC_DIM; ++src_dim)
    for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim) transform[tgt_dim][src_dim] = 0;

  if (SRC_DIM == TGT_DIM)
    for (int32_t src_dim = 0; src_dim < SRC_DIM; ++src_dim) {
      if ((mask & (0x01 << src_dim)) > 0) transform[src_dim][src_dim] = 1;
    }
  else
    for (int32_t src_dim = 0, tgt_dim = 0; src_dim < SRC_DIM; ++src_dim)
      if ((mask & (0x01 << src_dim)) > 0) transform[tgt_dim++][src_dim] = 1;

  return transform;
}

struct create_reduction_functor_fn {
  template <int32_t SRC_DIM,
            int32_t TGT_DIM,
            std::enable_if_t<SRC_DIM >= TGT_DIM && SRC_DIM >= 2>* = nullptr>
  void operator()(Runtime* runtime, const LegateContext& context, uint32_t mask)
  {
    auto proj_id      = context.get_projection_id(get_reduction_functor_id(SRC_DIM, TGT_DIM, mask));
    auto functor      = new ReductionFunctor<SRC_DIM, TGT_DIM>(runtime, mask);
    functors[proj_id] = functor;
    runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
  }

  template <int32_t SRC_DIM,
            int32_t TGT_DIM,
            std::enable_if_t<!(SRC_DIM >= TGT_DIM && SRC_DIM >= 2)>* = nullptr>
  void operator()(Runtime* runtime, const LegateContext& context, uint32_t mask)
  {
    assert(false);
  }
};

static uint32_t FACTORIALS[] = {
  0,
  1,
  2,
  6,
  24,
  120,
  720,
  40320,
  362880,
};

static uint32_t factorial(uint32_t val)
{
  if (val > 9) {
    uint32_t prod = 1;
    while (val > 0) prod *= val--;
    return prod;
  } else
    return FACTORIALS[val];
}

static std::vector<int32_t> convert_factoradic_to_sequence(int32_t dim, int32_t factoradic)
{
  int32_t orig = factoradic;
  std::vector<int32_t> digits(dim, 0);
  int32_t base = 2;
  for (int32_t idx = dim - 2; idx >= 0; --idx) {
    digits[idx] = factoradic % base;
    factoradic  = factoradic / base++;
  }

  std::list<int32_t> picked(dim);
  std::iota(picked.begin(), picked.end(), 0);

  std::vector<int32_t> sequence(dim, 0);
  for (int32_t idx = 0; idx < dim; ++idx) {
    auto it = picked.begin();
    for (int32_t cnt = 0; cnt < digits[idx]; ++cnt, ++it)
      ;
    sequence[idx] = *it;
    picked.erase(it);
  }
  return std::move(sequence);
}

// This family of functors reorder dimensions in a space
template <int32_t DIM>
class TransposeFunctor : public LegateProjectionFunctor {
 public:
  TransposeFunctor(Runtime* runtime, uint32_t id_in_factoradic);

 public:
  virtual DomainPoint project_point(const DomainPoint& point,
                                    const Domain& launch_domain) const override
  {
    return DomainPoint(transform_ * Point<DIM>(point));
  }

 public:
  static Transform<DIM, DIM> create_transform(uint32_t id_in_factoradic);

 private:
  const Transform<DIM, DIM> transform_;
};

template <int32_t DIM>
TransposeFunctor<DIM>::TransposeFunctor(Runtime* runtime, uint32_t id_in_factoradic)
  : LegateProjectionFunctor(runtime), transform_(create_transform(id_in_factoradic))
{
}

template <int32_t DIM>
/*static*/ Transform<DIM, DIM> TransposeFunctor<DIM>::create_transform(uint32_t id_in_factoradic)
{
  Transform<DIM, DIM> transform;

  for (int32_t src_dim = 0; src_dim < DIM; ++src_dim)
    for (int32_t tgt_dim = 0; tgt_dim < DIM; ++tgt_dim) transform[tgt_dim][src_dim] = 0;

  auto dims = convert_factoradic_to_sequence(DIM, id_in_factoradic);

  for (int32_t src_dim = 0; src_dim < DIM; ++src_dim) {
    auto tgt_dim                = dims[src_dim];
    transform[tgt_dim][src_dim] = 1;
  }

  return transform;
}

struct create_transpose_functor_fn {
  template <int32_t DIM>
  void operator()(Runtime* runtime, const LegateContext& context, uint32_t id_in_factoradic)
  {
    auto proj_id = context.get_projection_id(get_transpose_functor_id(DIM, id_in_factoradic));
    auto functor = new TransposeFunctor<DIM>(runtime, id_in_factoradic);
    assert(functors.find(proj_id) == functors.end());
    functors[proj_id] = functor;
    runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
  }
};

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const LegateContext& context)
{
  // Register reduction functors
  for (uint32_t src_dim = 2; src_dim <= LEGION_MAX_DIM; ++src_dim) {
    uint32_t num_masks = 1 << src_dim;
    for (uint32_t mask = 1; mask < num_masks; ++mask) {
      uint32_t mask_size = __builtin_popcount(mask);
      if (mask_size == src_dim) continue;
      for (uint32_t tgt_dim = src_dim; tgt_dim >= mask_size; --tgt_dim)
        double_dispatch(src_dim, tgt_dim, create_reduction_functor_fn{}, runtime, context, mask);
    }
  }

  // Register transpose functors
  for (uint32_t src_dim = 2; src_dim <= LEGION_MAX_DIM; ++src_dim) {
    uint32_t num_permutations = factorial(src_dim);
    for (uint32_t id_in_factoradic = 1; id_in_factoradic < num_permutations; ++id_in_factoradic)
      dim_dispatch(src_dim, create_transpose_functor_fn{}, runtime, context, id_in_factoradic);
  }
}

/*static*/ LegateProjectionFunctor* Core::get_projection_functor(ProjectionID functor_id)
{
  return functors[functor_id];
}

}  // namespace legate
