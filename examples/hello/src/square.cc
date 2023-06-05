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

#include "hello_world.h"
#include "legate_library.h"

namespace hello {

class SquareTask : public Task<SquareTask, SQUARE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    legate::Store& output        = context.outputs().at(0);
    legate::Rect<1> output_shape = output.shape<1>();
    auto out                     = output.write_accessor<float, 1>();

    legate::Store& input        = context.inputs().at(0);
    legate::Rect<1> input_shape = input.shape<1>();  // should be a 1-Dim array
    auto in                     = input.read_accessor<float, 1>();

    assert(input_shape == output_shape);

    logger.info() << "Elementwise square [" << output_shape.lo << "," << output_shape.hi << "]";

    // i is a global index for the complete array
    for (size_t i = input_shape.lo; i <= input_shape.hi; ++i) { out[i] = in[i] * in[i]; }
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::SquareTask::register_variants();
}

}  // namespace
