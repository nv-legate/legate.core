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

#include "util.h"

namespace fs = std::filesystem;

namespace legateio {

std::filesystem::path get_unique_path_for_task_index(legate::DomainPoint& task_index,
                                                     const std::string& dirname)
{
  std::stringstream ss;
  for (int32_t idx = 0; idx < task_index.dim; ++idx) {
    if (idx != 0) ss << ".";
    ss << task_index[idx];
  }
  auto filename = ss.str();

  return fs::path(dirname) / filename;
}

}  // namespace legateio
