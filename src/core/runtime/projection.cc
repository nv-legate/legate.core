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

#include "core/runtime/context.h"
#include "core/utilities/dispatch.h"
#include "legate_defines.h"

using namespace Legion;

namespace legate {

extern Logger log_legate;

// This special functor overrides the default projection implementation because it needs
// to know the the target color space for delinearization. Also note that this functor's
// project_point passes through input points, as we already know they are always 1D points
// and the output will be linearized back to integers.
class DelinearizationFunctor : public LegateProjectionFunctor {
 public:
  DelinearizationFunctor(Runtime* runtime);

 public:
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        const Legion::DomainPoint& point,
                                        const Legion::Domain& launch_domain) override;

 public:
  virtual Legion::DomainPoint project_point(const Legion::DomainPoint& point,
                                            const Legion::Domain& launch_domain) const override;
};

template <int32_t SRC_DIM, int32_t TGT_DIM>
class AffineFunctor : public LegateProjectionFunctor {
 public:
  AffineFunctor(Runtime* runtime, int32_t* dims, int32_t* weights, int32_t* offsets);

 public:
  DomainPoint project_point(const DomainPoint& point, const Domain& launch_domain) const override;

 public:
  static Legion::Transform<TGT_DIM, SRC_DIM> create_transform(int32_t* dims, int32_t* weights);

 private:
  const Legion::Transform<TGT_DIM, SRC_DIM> transform_;
  Point<TGT_DIM> offsets_;
};

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

DelinearizationFunctor::DelinearizationFunctor(Runtime* runtime) : LegateProjectionFunctor(runtime)
{
}

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

Legion::DomainPoint DelinearizationFunctor::project_point(const Legion::DomainPoint& point,
                                                          const Legion::Domain& launch_domain) const
{
  return point;
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
AffineFunctor<SRC_DIM, TGT_DIM>::AffineFunctor(Runtime* runtime,
                                               int32_t* dims,
                                               int32_t* weights,
                                               int32_t* offsets)
  : LegateProjectionFunctor(runtime), transform_(create_transform(dims, weights))
{
  for (int32_t dim = 0; dim < TGT_DIM; ++dim) offsets_[dim] = offsets[dim];

  // mapping to a different dimension
  if (SRC_DIM > TGT_DIM) {
    set_collective();
    return;
  }

  // find if there is `-1` in the dimensions
  std::set<int32_t> unique;
  for (int32_t dim = 0; dim < SRC_DIM; ++dim) {
    if (dims[dim] == -1) {
      set_collective();
      return;
    }
    unique.insert(dims[dim]);
  }
  // if there are repeated dimensions
  if (unique.size() != SRC_DIM) set_collective();
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
DomainPoint AffineFunctor<SRC_DIM, TGT_DIM>::project_point(const DomainPoint& point,
                                                           const Domain& launch_domain) const
{
  return DomainPoint(transform_ * Point<SRC_DIM>(point) + offsets_);
}

template <int32_t SRC_DIM, int32_t TGT_DIM>
/*static*/ Legion::Transform<TGT_DIM, SRC_DIM> AffineFunctor<SRC_DIM, TGT_DIM>::create_transform(
  int32_t* dims, int32_t* weights)
{
  Legion::Transform<TGT_DIM, SRC_DIM> transform;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim)
    for (int32_t src_dim = 0; src_dim < SRC_DIM; ++src_dim) transform[tgt_dim][src_dim] = 0;

  for (int32_t tgt_dim = 0; tgt_dim < TGT_DIM; ++tgt_dim) {
    int32_t src_dim = dims[tgt_dim];
    if (src_dim != -1) transform[tgt_dim][src_dim] = weights[tgt_dim];
  }

  return transform;
}

struct IdentityFunctor : public LegateProjectionFunctor {
  IdentityFunctor(Runtime* runtime) : LegateProjectionFunctor(runtime) {}
  DomainPoint project_point(const DomainPoint& point, const Domain&) const override
  {
    return point;
  }
};

static LegateProjectionFunctor* identity_functor{nullptr};
static std::unordered_map<ProjectionID, LegateProjectionFunctor*> functor_table{};
static std::mutex functor_table_lock{};

struct create_affine_functor_fn {
  static void spec_to_string(std::stringstream& ss,
                             int32_t src_ndim,
                             int32_t tgt_ndim,
                             int32_t* dims,
                             int32_t* weights,
                             int32_t* offsets)
  {
    ss << "\\(";
    for (int32_t idx = 0; idx < src_ndim; ++idx) {
      if (idx != 0) ss << ",";
      ss << "x" << idx;
    }
    ss << ")->(";
    for (int32_t idx = 0; idx < tgt_ndim; ++idx) {
      if (idx != 0) ss << ",";
      auto dim    = dims[idx];
      auto weight = weights[idx];
      auto offset = offsets[idx];
      if (dim != -1)
        if (weight != 0) {
          assert(dim != -1);
          if (weight != 1) ss << weight << "*";
          ss << "x" << dim;
        }
      if (offset != 0) {
        if (offset > 0)
          ss << "+" << offset;
        else
          ss << "-" << -offset;
      } else if (weight == 0)
        ss << "0";
    }
    ss << ")";
  }

  template <int32_t SRC_DIM, int32_t TGT_DIM>
  void operator()(
    Runtime* runtime, int32_t* dims, int32_t* weights, int32_t* offsets, ProjectionID proj_id)
  {
    auto functor = new AffineFunctor<SRC_DIM, TGT_DIM>(runtime, dims, weights, offsets);
#ifdef DEBUG_LEGATE
    std::stringstream ss;
    ss << "Register projection functor: functor: " << functor << ", id: " << proj_id << ", ";
    spec_to_string(ss, SRC_DIM, TGT_DIM, dims, weights, offsets);
    log_legate.debug() << ss.str();
#else
    log_legate.debug("Register projection functor: functor: %p, id: %d", functor, proj_id);
#endif
    runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);

    const std::lock_guard<std::mutex> lock(functor_table_lock);
    functor_table[proj_id] = functor;
  }
};

void register_legate_core_projection_functors(Legion::Runtime* runtime,
                                              const LibraryContext& context)
{
  auto proj_id = context.get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
  auto functor = new DelinearizationFunctor(runtime);
  log_legate.debug("Register delinearizing functor: functor: %p, id: %d", functor, proj_id);
  runtime->register_projection_functor(proj_id, functor, true /*silence warnings*/);
  {
    const std::lock_guard<std::mutex> lock(functor_table_lock);
    functor_table[proj_id] = functor;
  }
  identity_functor = new IdentityFunctor(runtime);
}

LegateProjectionFunctor* find_legate_projection_functor(ProjectionID proj_id)
{
  if (0 == proj_id) return identity_functor;
  const std::lock_guard<std::mutex> lock(functor_table_lock);
  auto result = functor_table[proj_id];
  if (nullptr == result) {
    log_legate.debug("Failed to find projection functor of id %d", proj_id);
    LEGATE_ABORT;
  }
  return result;
}

struct LinearizingPointTransformFunctor : public PointTransformFunctor {
  // This is actually an invertible functor, but we will not use this for inversion
  virtual bool is_invertible(void) const { return false; }

  virtual DomainPoint transform_point(const DomainPoint& point,
                                      const Domain& domain,
                                      const Domain& range)
  {
    assert(range.dim == 1);
    DomainPoint result;
    result.dim = 1;

    int32_t ndim = domain.dim;
    int64_t idx  = point[0];
    for (int32_t dim = 1; dim < ndim; ++dim) {
      int64_t extent = domain.rect_data[dim + ndim] - domain.rect_data[dim] + 1;
      idx            = idx * extent + point[dim];
    }
    result[0] = idx;
    return result;
  }
};

static auto* linearizing_point_transform_functor = new LinearizingPointTransformFunctor();

}  // namespace legate

extern "C" {

void legate_register_affine_projection_functor(int32_t src_ndim,
                                               int32_t tgt_ndim,
                                               int32_t* dims,
                                               int32_t* weights,
                                               int32_t* offsets,
                                               legion_projection_id_t proj_id)
{
  auto runtime = Runtime::get_runtime();
  legate::double_dispatch(src_ndim,
                          tgt_ndim,
                          legate::create_affine_functor_fn{},
                          runtime,
                          dims,
                          weights,
                          offsets,
                          proj_id);
}

void* legate_linearizing_point_transform_functor()
{
  return legate::linearizing_point_transform_functor;
}
}
