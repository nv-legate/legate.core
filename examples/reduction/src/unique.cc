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

#include <unordered_set>

#include "legate_library.h"
#include "reduction_cffi.h"

#include "core/utilities/dispatch.h"

namespace reduction {

namespace {

template <typename VAL>
void add_to_set(std::unordered_set<VAL>& unique_values, legate::Store& input)
{
  auto shape = input.shape<1>();
  if (shape.empty()) return;
  auto acc = input.read_accessor<VAL, 1>();
  for (legate::PointInRectIterator<1> it(shape, false /*fortran_order*/); it.valid(); ++it)
    unique_values.insert(acc[*it]);
}

template <typename VAL>
void copy_to_output(legate::Store& output, const std::unordered_set<VAL>& values)
{
  if (values.empty()) {
    output.bind_empty_data();
    return;
  }

  int64_t num_values = values.size();
  auto out_buf =
    output.create_output_buffer<VAL, 1>(legate::Point<1>(num_values), true /*bind_buffer*/);
  int64_t idx = 0;
  for (const auto& value : values) out_buf[idx++] = value;
}

template <legate::Type::Code CODE>
constexpr bool is_supported =
  !(legate::is_floating_point<CODE>::value || legate::is_complex<CODE>::value);
struct unique_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::Store& output, std::vector<legate::Store>& inputs)
  {
    using VAL = legate::legate_type_of<CODE>;

    std::unordered_set<VAL> unique_values;
    // Find unique values across all inputs
    for (auto& input : inputs) add_to_set(unique_values, input);
    // Copy the set of unique values to the output store
    copy_to_output(output, unique_values);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::Store& output, std::vector<legate::Store>& inputs)
  {
    LEGATE_ABORT;
  }
};

}  // namespace

class UniqueTask : public Task<UniqueTask, UNIQUE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs = context.inputs();
    auto& output = context.outputs().at(0);
    legate::type_dispatch(output.code(), unique_fn{}, output, inputs);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::UniqueTask::register_variants();
}

}  // namespace
