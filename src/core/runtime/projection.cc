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

#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "core/runtime/projection.h"
#include "core/utilities/dispatch.h"

using namespace Legion;

namespace legate {

class DelinearizationFunctor : public ProjectionFunctor {
 public:
  DelinearizationFunctor(Runtime* runtime);

 public:
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        const Legion::DomainPoint& point,
                                        const Legion::Domain& launch_domain);

 public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return 0; }
};

DelinearizationFunctor::DelinearizationFunctor(Runtime* runtime) : ProjectionFunctor(runtime) {}

LogicalRegion DelinearizationFunctor::project(LogicalPartition upper_bound,
                                              const DomainPoint& point,
                                              const Domain& launch_domain)
{
  const auto color_space =
    runtime->get_index_partition_color_space(upper_bound.get_index_partition());

  assert(color_space.dense());
  assert(point.dim == 1);

  std::vector<int64_t> strides(color_space.dim, 1);
  for (int32_t dim = color_space.dim - 1; dim > 0; --dim) {
    auto extent = color_space.rect_data[dim + color_space.dim] - color_space.rect_data[dim] + 1;
    strides[dim - 1] = strides[dim] * extent;
  }

  DomainPoint delinearized;
  delinearized.dim = color_space.dim;
  int64_t value    = point[0];
  for (int32_t dim = 0; dim < color_space.dim; ++dim) {
    delinearized[dim] = value / strides[dim];
    value             = value % strides[dim];
  }

  if (runtime->has_logical_subregion_by_color(upper_bound, delinearized))
    return runtime->get_logical_subregion_by_color(upper_bound, delinearized);
  else
    return LogicalRegion::NO_REGION;
}

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const LibraryContext& context)
{
  auto proj_id = context.get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
  auto functor = new DelinearizationFunctor(runtime);
  runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
}

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

template <int32_t SRC_DIM, int32_t TGT_DIM>
class AffineFunctor : public LegateProjectionFunctor {
 public:
  AffineFunctor(Runtime* runtime, int32_t* dims, int32_t* offsets);

 public:
  virtual DomainPoint project_point(const DomainPoint& point,
                                    const Domain& launch_domain) const override
  {
    return DomainPoint(transform_ * Point<SRC_DIM>(point) + offsets_);
  }

 public:
  static Transform<TGT_DIM, SRC_DIM> create_transform(int32_t* dims);

 private:
  const Transform<TGT_DIM, SRC_DIM> transform_;
  Point<TGT_DIM> offsets_;
};

template <int32_t SRC_DIM, int32_t TGT_DIM>
AffineFunctor<SRC_DIM, TGT_DIM>::AffineFunctor(Runtime* runtime, int32_t* dims, int32_t* offsets)
  : LegateProjectionFunctor(runtime), transform_(create_transform(dims))
{
  for (int32_t dim = 0; dim < TGT_DIM; ++dim) offsets_[dim] = offsets[dim];
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
/*static*/ Transform<TGT_DIM, SRC_DIM> AffineFunctor<SRC_DIM, TGT_DIM>::create_transform(
  int32_t* dims)
{
  Transform<TGT_DIM, SRC_DIM> transform;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim)
    for (int32_t src_dim = 0; src_dim < SRC_DIM; ++src_dim) transform[tgt_dim][src_dim] = 0;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim) {
    int32_t src_dim = dims[tgt_dim];
    if (src_dim != -1) transform[tgt_dim][src_dim] = 1;
  }

  return transform;
}

static std::unordered_map<ProjectionID, LegateProjectionFunctor*> functor_table;
static std::mutex functor_table_lock;

struct create_affine_functor_fn {
  template <int32_t SRC_DIM, int32_t TGT_DIM>
  void operator()(Runtime* runtime, int32_t* dims, int32_t* offsets, ProjectionID proj_id)
  {
    auto functor = new AffineFunctor<SRC_DIM, TGT_DIM>(runtime, dims, offsets);
    runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);

    const std::lock_guard<std::mutex> lock(functor_table_lock);
    functor_table[proj_id] = functor;
  }
};

LegateProjectionFunctor* find_legate_projection_functor(ProjectionID proj_id)
{
  if (0 == proj_id) return nullptr;
  const std::lock_guard<std::mutex> lock(functor_table_lock);
  return functor_table[proj_id];
}

}  // namespace legate

extern "C" {

void legate_register_affine_projection_functor(int32_t src_ndim,
                                               int32_t tgt_ndim,
                                               int32_t* dims,
                                               int32_t* offsets,
                                               legion_projection_id_t proj_id)
{
  auto runtime = Runtime::get_runtime();
  legate::double_dispatch(
    src_ndim, tgt_ndim, legate::create_affine_functor_fn{}, runtime, dims, offsets, proj_id);
}
}
