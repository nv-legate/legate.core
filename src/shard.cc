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

#include "shard.h"

namespace legate {

using namespace Legion;

class LegateCoreShardingFunctor : public ShardingFunctor {
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

void register_legate_core_sharding_functors(Legion::Runtime* runtime, const LegateContext& context)
{
  auto sharding_id = context.get_sharding_id(0);
  runtime->register_sharding_functor(sharding_id, new LegateCoreShardingFunctor());
}

}  // namespace legate
