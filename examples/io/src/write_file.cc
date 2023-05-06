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

#include <fstream>

#include "legate_library.h"
#include "legateio.h"

#include "core/utilities/dispatch.h"

namespace legateio {

namespace {

struct write_fn {
  template <legate::Type::Code CODE>
  void operator()(const legate::Store& input, const std::string& filename)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape  = input.shape<1>();
    auto code   = input.code<int64_t>();
    size_t size = shape.volume();

    // Store the type code and the number of elements in the array at the beginning of the file
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&code), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    auto acc = input.read_accessor<VAL, 1>();
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      out.write(reinterpret_cast<const char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

class WriteFileTask : public Task<WriteFileTask, WRITE_FILE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto filename = context.scalars().at(0).value<std::string>();
    auto& input   = context.inputs().at(0);
    logger.print() << "Write to " << filename;

    legate::type_dispatch(input.code(), write_fn{}, input, filename);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteFileTask::register_variants();
}

}  // namespace
