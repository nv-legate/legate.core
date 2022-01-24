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
#include <unordered_map>

#include "legate.h"

#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/utilities/linearize.h"

using namespace Legion;

namespace legate {

static std::unordered_map<ProjectionID, ShardID> functor_id_table;
static std::mutex functor_table_lock;

class ToplevelTaskShardingFunctor : public ShardingFunctor {
 public:
  virtual ShardID shard(const DomainPoint& p, const Domain& launch_space, const size_t total_shards)
  {
    // Just tile this space in 1D
    const Point<1> point = p;
    const Rect<1> space  = launch_space;
    const size_t size    = (space.hi[0] - space.lo[0]) + 1;
    const size_t chunk   = (size + total_shards - 1) / total_shards;
    return (point[0] - space.lo[0]) / chunk;
  }
};

class LinearizingFunctor : public ShardingFunctor {
 public:
  virtual ShardID shard(const DomainPoint& p, const Domain& launch_space, const size_t total_shards)
  {
    const size_t size  = launch_space.get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    return linearize(launch_space.lo(), launch_space.hi(), p) / chunk;
  }

  virtual bool is_invertible(void) const { return true; }

  virtual void invert(ShardID shard,
                      const Domain& shard_domain,
                      const Domain& full_domain,
                      const size_t total_shards,
                      std::vector<DomainPoint>& points)
  {
    assert(shard_domain == full_domain);
    const size_t size  = shard_domain.get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    size_t idx         = shard * chunk;
    size_t lim         = std::min((shard + 1) * chunk, size);
    if (idx >= lim) return;
    DomainPoint point = delinearize(shard_domain.lo(), shard_domain.hi(), idx);
    for (; idx < lim; ++idx) {
      points.push_back(point);
      for (int dim = shard_domain.dim - 1; dim >= 0; --dim) {
        if (point[dim] < shard_domain.hi()[dim]) {
          point[dim]++;
          break;
        }
        point[dim] = shard_domain.lo()[dim];
      }
    }
  }
};

void register_legate_core_sharding_functors(Legion::Runtime* runtime, const LibraryContext& context)
{
  runtime->register_sharding_functor(context.get_sharding_id(LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID),
                                     new ToplevelTaskShardingFunctor());

  auto sharding_id = context.get_sharding_id(LEGATE_CORE_LINEARIZE_SHARD_ID);
  runtime->register_sharding_functor(sharding_id, new LinearizingFunctor());
  // Use linearizing functor for identity projections
  functor_id_table[0] = sharding_id;
  // and for the delinearizing projection
  functor_id_table[context.get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID)] = sharding_id;
}

class TilingFunctor : public ShardingFunctor {
 public:
  TilingFunctor(const mapping::Grid& proc_grid, const mapping::Grid& shard_grid)
    : proc_grid_(proc_grid), shard_grid_(shard_grid)
  {
  }

 public:
  virtual ShardID shard(const DomainPoint& p, const Domain& launch_space, const size_t total_shards)
  {
    DomainPoint pt = p;
    auto ndim      = pt.dim;
    for (int32_t dim = 0; dim < ndim; ++dim)
      pt[dim] = (pt[dim] / proc_grid_.grid[dim]) % shard_grid_.grid[dim];

    int32_t shard_id = 0;
    for (int32_t dim = 0; dim < ndim; ++dim) shard_id += pt[dim] * shard_grid_.pitches[dim];
    assert(0 <= shard_id && static_cast<size_t>(shard_id) < total_shards);

    // if (ndim == 2)
    //   fprintf(stderr, "( %ld %ld ) (%ld %ld) --> %d\n", p[0], p[1], pt[0], pt[1], shard_id);
    return shard_id;
  }

 private:
  mapping::Grid proc_grid_;
  mapping::Grid shard_grid_;
  int32_t num_procs_;
};

void register_new_tiling_functor(Legion::Runtime* runtime,
                                 Legion::ShardingID sharding_id,
                                 const mapping::Grid& proc_grid,
                                 const mapping::Grid& shard_grid)
{
  runtime->register_sharding_functor(sharding_id, new TilingFunctor(proc_grid, shard_grid));
}

class LegateShardingFunctor : public ShardingFunctor {
 public:
  LegateShardingFunctor(LegateProjectionFunctor* proj_functor) : proj_functor_(proj_functor) {}

 public:
  virtual ShardID shard(const DomainPoint& p,
                        const Domain& launch_space,
                        const size_t total_shards) override
  {
    auto lo    = proj_functor_->project_point(launch_space.lo(), launch_space);
    auto hi    = proj_functor_->project_point(launch_space.hi(), launch_space);
    auto point = proj_functor_->project_point(p, launch_space);

    const size_t size  = Domain(lo, hi).get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    return linearize(lo, hi, point) / chunk;
  }

 private:
  LegateProjectionFunctor* proj_functor_;
};

ShardingID find_sharding_functor_by_projection_functor(Legion::ProjectionID proj_id)
{
  const std::lock_guard<std::mutex> lock(legate::functor_table_lock);
  assert(functor_id_table.find(proj_id) != functor_id_table.end());
  return functor_id_table[proj_id];
}

}  // namespace legate

extern "C" {

void legate_create_sharding_functor_using_projection(Legion::ShardID shard_id,
                                                     Legion::ProjectionID proj_id)
{
  auto runtime = Runtime::get_runtime();
  auto sharding_functor =
    new legate::LegateShardingFunctor(legate::find_legate_projection_functor(proj_id));
  runtime->register_sharding_functor(shard_id, sharding_functor, true /*silence warnings*/);
  const std::lock_guard<std::mutex> lock(legate::functor_table_lock);
  legate::functor_id_table[proj_id] = shard_id;
}
}
