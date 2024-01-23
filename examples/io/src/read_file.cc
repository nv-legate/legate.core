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

namespace legateio {

namespace {

struct read_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::Store& output,
                  const std::string& filename,
                  int64_t my_id,
                  int64_t num_readers)
  {
    using VAL = legate::legate_type_of<CODE>;

    int64_t code;
    size_t size;

    // All reader tasks need to read the header to figure out the total number of elements
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    in.read(reinterpret_cast<char*>(&code), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

    if (static_cast<legate::Type::Code>(code) != CODE) {
      logger.error() << "Type mismatch: " << CODE << " != " << code;
      LEGATE_ABORT;
    }

    // Compute the absolute offsets to the section that this reader task
    // is supposed to read from the file
    int64_t my_lo = my_id * size / num_readers;
    auto ssize    = static_cast<int64_t>(size);
    int64_t my_hi = std::min((my_id + 1) * ssize / num_readers, ssize);

    // Then compute the extent for the output and create the output buffer to populate
    int64_t my_ext = my_hi - my_lo;
    auto buf       = output.create_output_buffer<VAL, 1>(legate::Point<1>(my_ext));

    // Skip to the right offset where the data assigned to this reader task actually starts
    if (my_lo != 0) in.seekg(my_lo * sizeof(VAL), std::ios_base::cur);
    for (int64_t idx = 0; idx < my_ext; ++idx) {
      auto ptr = buf.ptr(legate::Point<1>(idx));
      in.read(reinterpret_cast<char*>(ptr), sizeof(VAL));
    }

    // Finally, bind the output buffer to the store
    //
    // Some minor details about unbound stores:
    //
    // 1) Though this example binds the buffer after it's populated, it's not strictly necessary.
    //    In fact, the create_output_buffer call takes an optional boolean argument that binds
    //    the created buffer immediately.
    // 2) The bind_data call takes the extents of data as an argument, just in case the buffer
    //    is actually bigger than extents of the actual data.
    //
    output.bind_data(buf, legate::Point<1>(my_ext));
  }
};

}  // namespace

class ReadFileTask : public Task<ReadFileTask, READ_FILE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto filename = context.scalars().at(0).value<std::string>();
    auto& output  = context.outputs().at(0);

    // The task context contains metadata about the launch so each reader task can figure out
    // which part of the file it needs to read into the output.
    int64_t my_id       = context.is_single_task() ? 0 : context.get_task_index()[0];
    int64_t num_readers = context.is_single_task() ? 1 : context.get_launch_domain().get_volume();
    logger.print() << "Read " << filename << " (" << my_id + 1 << "/" << num_readers << ")";

    // type_dispatch converts the first argument to a non-type template argument
    legate::type_dispatch(output.code(), read_fn{}, output, filename, my_id, num_readers);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::ReadFileTask::register_variants();
}

}  // namespace
