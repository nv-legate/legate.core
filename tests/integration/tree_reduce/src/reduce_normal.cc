/* Copyright 2023 NVIDIA Corporation
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

#include "reduce_normal.h"

namespace tree_reduce {

/*static*/ void ReduceNormalTask::cpu_variant(legate::TaskContext& context)
{
  auto& inputs = context.inputs();
  auto& output = context.outputs().at(0);
  for (auto& input : inputs) {
    auto shape = input.shape<1>();
    assert(shape.empty() || shape.volume() == TILE_SIZE);
  }
  output.create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
}

}  // namespace tree_reduce
