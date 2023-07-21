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

#include "library.h"
#include "tree_reduce_cffi.h"

namespace tree_reduce {

struct ProduceUnboundTask : public Task<ProduceUnboundTask, PRODUCE_UNBOUND> {
  static void cpu_variant(legate::TaskContext& context);
};

}  // namespace tree_reduce
