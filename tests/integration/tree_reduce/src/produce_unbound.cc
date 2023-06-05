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

#include "produce_unbound.h"

namespace tree_reduce {

/*static*/ void ProduceUnboundTask::cpu_variant(legate::TaskContext& context)
{
  auto& output = context.outputs().at(0);
  auto size    = context.get_task_index()[0] + 1;
  auto buffer  = output.create_output_buffer<int64_t, 1>(legate::Point<1>(size), true /*bind*/);
  for (int64_t idx = 0; idx < size; ++idx) buffer[idx] = size;
}

}  // namespace tree_reduce
