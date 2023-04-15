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

#include "library.h"
#include "registry_cffi.h"

namespace rg {

class HelloTask : public Task<HelloTask, HELLO> {
 public:
  static void cpu_variant(legate::TaskContext& context) { log_registry.info() << "Hello"; }
};

}  // namespace rg

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks() { rg::HelloTask::register_variants(); }

}  // namespace