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

#include "legate_library.h"
#include "reduction_cffi.h"

#include "core/utilities/dispatch.h"

namespace reduction {

class BincountTask : public Task<BincountTask, BINCOUNT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs().at(0);
    auto& output = context.reductions().at(0);

    auto in_shape  = input.shape<1>();
    auto out_shape = output.shape<1>();

    auto in_acc  = input.read_accessor<uint64_t, 1>();
    auto out_acc = output.reduce_accessor<legate::SumReduction<uint64_t>, true, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto& value = in_acc[*it];
      legate::Point<1> pos_reduce(static_cast<int64_t>(value));

      if (out_shape.contains(pos_reduce)) out_acc.reduce(pos_reduce, 1);
    }
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::BincountTask::register_variants();
}

}  // namespace
