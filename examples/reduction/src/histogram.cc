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

namespace {

struct histogram_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::Store& result, legate::Store& input, legate::Store& bins)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto in_shape  = input.shape<1>();
    auto bin_shape = bins.shape<1>();

    assert(!bin_shape.empty());
    if (in_shape.empty()) return;

    auto num_bins = bin_shape.hi[0] - bin_shape.lo[0];

    auto in_acc  = input.read_accessor<VAL, 1>();
    auto bin_acc = bins.read_accessor<VAL, 1>();
    auto res_acc = result.reduce_accessor<legate::SumReduction<uint64_t>, true, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto& value = in_acc[*it];
      // Use a naive algorithm that loops all bin edges to find a match
      for (auto bin_idx = 0; bin_idx < num_bins; ++bin_idx)
        if (bin_acc[bin_idx] <= value && value < bin_acc[bin_idx + 1]) {
          res_acc.reduce(bin_idx, 1);
          break;
        }
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::Store& result, legate::Store& input, legate::Store& bins)
  {
    assert(false);
  }
};

}  // namespace

class HistogramTask : public Task<HistogramTask, HISTOGRAM> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs().at(0);
    auto& bins   = context.inputs().at(1);
    auto& result = context.reductions().at(0);

    legate::type_dispatch(input.code(), histogram_fn{}, result, input, bins);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::HistogramTask::register_variants();
}

}  // namespace
