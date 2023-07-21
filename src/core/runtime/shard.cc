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

namespace legate {

static std::unordered_map<Legion::ProjectionID, Legion::ShardID> functor_id_table;
static std::mutex functor_table_lock;

class ToplevelTaskShardingFunctor : public Legion::ShardingFunctor {
 public:
  virtual Legion::ShardID shard(const DomainPoint& p,
                                const Domain& launch_space,
                                const size_t total_shards)
  {
    // Just tile this space in 1D
    const Point<1> point = p;
    const Rect<1> space  = launch_space;
    const size_t size    = (space.hi[0] - space.lo[0]) + 1;
    const size_t chunk   = (size + total_shards - 1) / total_shards;
    return (point[0] - space.lo[0]) / chunk;
  }
};

class LinearizingShardingFunctor : public Legion::ShardingFunctor {
 public:
  virtual Legion::ShardID shard(const DomainPoint& p,
                                const Domain& launch_space,
                                const size_t total_shards)
  {
    const size_t size  = launch_space.get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    return linearize(launch_space.lo(), launch_space.hi(), p) / chunk;
  }

  virtual bool is_invertible(void) const { return true; }

  virtual void invert(Legion::ShardID shard,
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

void register_legate_core_sharding_functors(Legion::Runtime* runtime, const LibraryContext* context)
{
  runtime->register_sharding_functor(context->get_sharding_id(LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID),
                                     new ToplevelTaskShardingFunctor(),
                                     true /*silence warnings*/);

  auto sharding_id = context->get_sharding_id(LEGATE_CORE_LINEARIZE_SHARD_ID);
  runtime->register_sharding_functor(
    sharding_id, new LinearizingShardingFunctor(), true /*silence warnings*/);
  // Use linearizing functor for identity projections
  functor_id_table[0] = sharding_id;
  // and for the delinearizing projection
  functor_id_table[context->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID)] = sharding_id;
}

class LegateShardingFunctor : public Legion::ShardingFunctor {
 public:
  LegateShardingFunctor(LegateProjectionFunctor* proj_functor,
                        uint32_t start_node,
                        uint32_t end_node,
                        uint32_t offset,
                        uint32_t per_node_count)
    : proj_functor_(proj_functor),
      start_node_id_(start_node),
      end_node_id_(end_node),
      offset_(offset),
      per_node_count_(per_node_count)
  {
  }

 public:
  virtual Legion::ShardID shard(const DomainPoint& p,
                                const Domain& launch_space,
                                const size_t total_shards) override
  {
    auto lo    = proj_functor_->project_point(launch_space.lo(), launch_space);
    auto hi    = proj_functor_->project_point(launch_space.hi(), launch_space);
    auto point = proj_functor_->project_point(p, launch_space);

    auto shard_id = (linearize(lo, hi, point) + offset_) / per_node_count_ + start_node_id_;
#ifdef DEBUG_LEGATE
    assert(start_node_id_ <= shard_id && shard_id < end_node_id_);
#endif
    return shard_id;
  }

 private:
  LegateProjectionFunctor* proj_functor_;
  uint32_t start_node_id_;
  uint32_t end_node_id_;
  uint32_t offset_;
  uint32_t per_node_count_;
};

Legion::ShardingID find_sharding_functor_by_projection_functor(Legion::ProjectionID proj_id)
{
  const std::lock_guard<std::mutex> lock(legate::functor_table_lock);
  assert(functor_id_table.find(proj_id) != functor_id_table.end());
  return functor_id_table[proj_id];
}

struct ShardingCallbackArgs {
  Legion::ShardID shard_id;
  Legion::ProjectionID proj_id;
  uint32_t start_node;
  uint32_t end_node;
  uint32_t offset;
  uint32_t per_node_count;
};

static void sharding_functor_registration_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto p_args = static_cast<ShardingCallbackArgs*>(args.buffer.get_ptr());

  auto runtime = Legion::Runtime::get_runtime();
  auto sharding_functor =
    new legate::LegateShardingFunctor(legate::find_legate_projection_functor(p_args->proj_id),
                                      p_args->start_node,
                                      p_args->end_node,
                                      p_args->offset,
                                      p_args->per_node_count);
  runtime->register_sharding_functor(p_args->shard_id, sharding_functor, true /*silence warnings*/);
}

}  // namespace legate

extern "C" {

void legate_create_sharding_functor_using_projection(Legion::ShardID shard_id,
                                                     Legion::ProjectionID proj_id,
                                                     uint32_t start_node,
                                                     uint32_t end_node,
                                                     uint32_t offset,
                                                     uint32_t per_node_count)
{
  auto runtime = Legion::Runtime::get_runtime();
  legate::ShardingCallbackArgs args{
    shard_id, proj_id, start_node, end_node, offset, per_node_count};
  {
    const std::lock_guard<std::mutex> lock(legate::functor_table_lock);
    legate::functor_id_table[proj_id] = shard_id;
  }
  Legion::UntypedBuffer buffer(&args, sizeof(args));
  Legion::Runtime::perform_registration_callback(
    legate::sharding_functor_registration_callback, buffer, false /*global*/, false /*dedup*/);
}
}
