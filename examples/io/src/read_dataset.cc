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

namespace detail {

struct read_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(legate::Store& output, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    std::ifstream in(path, std::ios::binary | std::ios::in);

    legate::Point<DIM> extents;
    for (int32_t idx = 0; idx < DIM; ++idx)
      in.read(reinterpret_cast<char*>(&extents[idx]), sizeof(legate::coord_t));

    logger.print() << "Read a sub-array of extents " << extents << " from " << path;

    auto buf = output.create_output_buffer<VAL, DIM>(extents);
    legate::Rect<DIM> shape(legate::Point<DIM>::ZEROES(), extents - legate::Point<DIM>::ONES());
    for (legate::PointInRectIterator it(shape); it.valid(); ++it) {
      auto ptr = buf.ptr(*it);
      in.read(reinterpret_cast<char*>(ptr), sizeof(VAL));
    }

    output.bind_data(buf, extents);
  }
};

}  // namespace detail

class ReadDatasetTask : public Task<ReadDatasetTask, READ_DATASET> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto dirname = context.scalars()[0].value<std::string>();
    auto& output = context.outputs()[0];

    auto task_index = context.get_task_index();
    if (context.is_single_task()) {
      task_index     = legate::DomainPoint();
      task_index.dim = output.dim();
    }

    auto path = get_unique_path_for_task_index(task_index, dirname);

    legate::double_dispatch(output.dim(), output.code(), detail::read_fn{}, output, path);
  }
};

}  // namespace legateio

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks()
{
  legateio::ReadDatasetTask::register_variants();
}

}  // namespace
