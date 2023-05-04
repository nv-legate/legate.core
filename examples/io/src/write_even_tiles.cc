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

#include <filesystem>
#include <fstream>

#include "legate_library.h"
#include "legateio.h"
#include "util.h"

#include "core/utilities/span.h"

namespace fs = std::filesystem;

namespace legateio {

namespace {

void write_header(std::ofstream& out,
                  legate::Type::Code type_code,
                  const legate::Span<const int32_t>& shape,
                  const legate::Span<const int32_t>& tile_shape)
{
  assert(shape.size() == tile_shape.size());
  int32_t dim = shape.size();
  // Dump the type code, the array's shape and the tile shape to the header
  out.write(reinterpret_cast<const char*>(&type_code), sizeof(int32_t));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
  for (auto& v : shape) out.write(reinterpret_cast<const char*>(&v), sizeof(int32_t));
  for (auto& v : tile_shape) out.write(reinterpret_cast<const char*>(&v), sizeof(int32_t));
}

}  // namespace

class WriteEvenTilesTask : public Task<WriteEvenTilesTask, WRITE_EVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto dirname                           = context.scalars().at(0).value<std::string>();
    legate::Span<const int32_t> shape      = context.scalars().at(1).values<int32_t>();
    legate::Span<const int32_t> tile_shape = context.scalars().at(2).values<int32_t>();
    auto& input                            = context.inputs().at(0);

    auto launch_domain = context.get_launch_domain();
    auto task_index    = context.get_task_index();
    auto is_first_task = context.is_single_task() || task_index == launch_domain.lo();

    if (is_first_task) {
      auto header = fs::path(dirname) / ".header";
      logger.print() << "Write to " << header;
      std::ofstream out(header, std::ios::binary | std::ios::out | std::ios::trunc);
      write_header(out, input.code(), shape, tile_shape);
    }

    write_to_file(context, dirname, input);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteEvenTilesTask::register_variants();
}

}  // namespace
