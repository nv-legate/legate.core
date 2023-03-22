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

#include "core/utilities/dispatch.h"

namespace fs = std::filesystem;

namespace legateio {

namespace detail {

struct header_write_fn {
  template <int32_t DIM>
  void operator()(std::ofstream& out,
                  const legate::Domain& launch_domain,
                  legate::LegateTypeCode type_code)
  {
    legate::Rect<DIM> rect(launch_domain);
    auto extents = rect.hi - rect.lo + legate::Point<DIM>::ONES();

    out.write(reinterpret_cast<const char*>(&type_code), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&launch_domain.dim), sizeof(int32_t));
    for (int32_t idx = 0; idx < DIM; ++idx)
      out.write(reinterpret_cast<const char*>(&extents[idx]), sizeof(legate::coord_t));
  }
};

struct write_fn {
  template <legate::LegateTypeCode CODE, int32_t DIM>
  void operator()(const legate::Store& input, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = input.shape<DIM>();
    auto empty = shape.empty();
    auto extents =
      empty ? legate::Point<DIM>::ZEROES() : shape.hi - shape.lo + legate::Point<DIM>::ONES();

    int32_t dim  = DIM;
    int32_t code = input.code<int32_t>();

    logger.print() << "Write a sub-array " << shape << " to " << path;

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    for (int32_t idx = 0; idx < DIM; ++idx)
      out.write(reinterpret_cast<const char*>(&extents[idx]), sizeof(legate::coord_t));

    if (empty) return;
    auto acc = input.read_accessor<VAL, DIM>();
    for (legate::PointInRectIterator it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      out.write(reinterpret_cast<const char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace detail

class WriteUnevenTilesTask : public Task<WriteUnevenTilesTask, WRITE_UNEVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto dirname = context.scalars()[0].value<std::string>();
    auto& input  = context.inputs()[0];

    auto launch_domain = context.get_launch_domain();
    auto task_index    = context.get_task_index();
    auto is_first_task = context.is_single_task() || task_index == launch_domain.lo();
    if (context.is_single_task()) {
      launch_domain     = legate::Domain();
      task_index        = legate::DomainPoint();
      launch_domain.dim = input.dim();
      task_index.dim    = input.dim();
    }

    if (is_first_task) {
      auto header = fs::path(dirname) / ".header";
      logger.print() << "Write to " << header;
      std::ofstream out(header, std::ios::binary | std::ios::out | std::ios::trunc);
      legate::dim_dispatch(
        launch_domain.dim, detail::header_write_fn{}, out, launch_domain, input.code());
    }

    auto path = get_unique_path_for_task_index(task_index, dirname);
    legate::double_dispatch(input.dim(), input.code(), detail::write_fn{}, input, path);
  }
};

}  // namespace legateio

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteUnevenTilesTask::register_variants();
}

}  // namespace
