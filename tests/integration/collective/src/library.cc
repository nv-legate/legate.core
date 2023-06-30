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
#include "collective_test.h"

namespace collective {

static const char* const library_name = "collective";

Legion::Logger log_collective(library_name);

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  CollectiveTestTask::register_variants(context);
}

}  // namespace collective

extern "C" {

void collective_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  legate::Core::perform_registration<collective::registration_callback>();
}
}
