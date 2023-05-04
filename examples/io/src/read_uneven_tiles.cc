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

namespace fs = std::filesystem;

namespace legateio {

namespace {

struct read_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::Store& output, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    std::ifstream in(path, std::ios::binary | std::ios::in);

    // Read the header of each file to extract the extents
    legate::Point<DIM> extents;
    for (int32_t idx = 0; idx < DIM; ++idx)
      in.read(reinterpret_cast<char*>(&extents[idx]), sizeof(legate::coord_t));

    logger.print() << "Read a sub-array of extents " << extents << " from " << path;

    // Use the extents to create an output buffer
    auto buf = output.create_output_buffer<VAL, DIM>(extents);
    legate::Rect<DIM> shape(legate::Point<DIM>::ZEROES(), extents - legate::Point<DIM>::ONES());
    if (!shape.empty())
      // Read the file data. The iteration order here should be the same as in the writer task
      for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
        auto ptr = buf.ptr(*it);
        in.read(reinterpret_cast<char*>(ptr), sizeof(VAL));
      }

    // Finally, bind the output buffer to the store
    output.bind_data(buf, extents);
  }
};

}  // namespace

class ReadUnevenTilesTask : public Task<ReadUnevenTilesTask, READ_UNEVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto dirname = context.scalars().at(0).value<std::string>();
    auto& output = context.outputs().at(0);

    auto path = get_unique_path_for_task_index(context, output.dim(), dirname);
    // double_dispatch converts the first two arguments to non-type template arguments
    legate::double_dispatch(output.dim(), output.code(), read_fn{}, output, path);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::ReadUnevenTilesTask::register_variants();
}

}  // namespace
