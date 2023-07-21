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

namespace {

struct header_write_fn {
  template <int32_t DIM>
  void operator()(std::ofstream& out,
                  const legate::Domain& launch_domain,
                  legate::Type::Code type_code)
  {
    legate::Rect<DIM> rect(launch_domain);
    auto extents = rect.hi - rect.lo + legate::Point<DIM>::ONES();

    // The header contains the type code and the launch shape
    out.write(reinterpret_cast<const char*>(&type_code), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&launch_domain.dim), sizeof(int32_t));
    for (int32_t idx = 0; idx < DIM; ++idx)
      out.write(reinterpret_cast<const char*>(&extents[idx]), sizeof(legate::coord_t));
  }
};

}  // namespace

class WriteUnevenTilesTask : public Task<WriteUnevenTilesTask, WRITE_UNEVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto dirname = context.scalars().at(0).value<std::string>();
    auto& input  = context.inputs().at(0);

    auto launch_domain = context.get_launch_domain();
    auto task_index    = context.get_task_index();
    auto is_first_task = context.is_single_task() || task_index == launch_domain.lo();

    // Only the first task needs to dump the header
    if (is_first_task) {
      // When the task is a single task, we create a launch domain of volume 1
      if (context.is_single_task()) {
        launch_domain     = legate::Domain();
        launch_domain.dim = input.dim();
      }

      auto header = fs::path(dirname) / ".header";
      logger.print() << "Write to " << header;
      std::ofstream out(header, std::ios::binary | std::ios::out | std::ios::trunc);
      legate::dim_dispatch(launch_domain.dim, header_write_fn{}, out, launch_domain, input.code());
    }

    write_to_file(context, dirname, input);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteUnevenTilesTask::register_variants();
}

}  // namespace
