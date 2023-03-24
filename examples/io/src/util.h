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

#pragma once

#include <filesystem>

#include "core/data/store.h"
#include "core/runtime/context.h"

namespace legateio {

std::filesystem::path get_unique_path_for_task_index(const legate::TaskContext& task_context,
                                                     int32_t ndim,
                                                     const std::string& dirname);

void write_to_file(legate::TaskContext& task_context,
                   const std::string& dirname,
                   const legate::Store& store);

}  // namespace legateio
