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

    legate::Rect<DIM> shape = output.shape<DIM>();

    if (shape.empty()) return;

    std::ifstream in(path, std::ios::binary | std::ios::in);

    legate::Point<DIM> extents;
    for (int32_t idx = 0; idx < DIM; ++idx)
      in.read(reinterpret_cast<char*>(&extents[idx]), sizeof(legate::coord_t));

    // Since the shape is already fixed on the Python side, the sub-store's extents should be the
    // same as what's stored in the file
    assert(shape.hi - shape.lo + legate::Point<DIM>::ONES() == extents);

    logger.print() << "Read a sub-array of rect " << shape << " from " << path;

    auto acc = output.write_accessor<VAL, DIM>();
    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      in.read(reinterpret_cast<char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

class ReadEvenTilesTask : public Task<ReadEvenTilesTask, READ_EVEN_TILES> {
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
  legateio::ReadEvenTilesTask::register_variants();
}

}  // namespace
