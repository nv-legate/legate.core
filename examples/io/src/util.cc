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

#include "legateio.h"
#include "util.h"

#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"

namespace fs = std::filesystem;

namespace legateio {

namespace {

struct write_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(const legate::Store& store, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = store.shape<DIM>();
    auto empty = shape.empty();
    auto extents =
      empty ? legate::Point<DIM>::ZEROES() : shape.hi - shape.lo + legate::Point<DIM>::ONES();

    int32_t dim  = DIM;
    int32_t code = store.code<int32_t>();

    logger.print() << "Write a sub-array " << shape << " to " << path;

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    // Each file for a chunk starts with the extents
    for (int32_t idx = 0; idx < DIM; ++idx)
      out.write(reinterpret_cast<const char*>(&extents[idx]), sizeof(legate::coord_t));

    if (empty) return;
    auto acc = store.read_accessor<VAL, DIM>();
    // The iteration order here should be consistent with that in the reader task, otherwise
    // the read data can be transposed.
    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      out.write(reinterpret_cast<const char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

std::filesystem::path get_unique_path_for_task_index(const legate::TaskContext& context,
                                                     int32_t ndim,
                                                     const std::string& dirname)
{
  auto task_index = context.get_task_index();
  // If this was a single task, we use (0, ..., 0) for the task index
  if (context.is_single_task()) {
    task_index     = legate::DomainPoint();
    task_index.dim = ndim;
  }

  std::stringstream ss;
  for (int32_t idx = 0; idx < task_index.dim; ++idx) {
    if (idx != 0) ss << ".";
    ss << task_index[idx];
  }
  auto filename = ss.str();

  return fs::path(dirname) / filename;
}

void write_to_file(legate::TaskContext& task_context,
                   const std::string& dirname,
                   const legate::Store& store)
{
  auto path = get_unique_path_for_task_index(task_context, store.dim(), dirname);
  // double_dispatch converts the first two arguments to non-type template arguments
  legate::double_dispatch(store.dim(), store.code(), write_fn{}, store, path);
}

}  // namespace legateio
