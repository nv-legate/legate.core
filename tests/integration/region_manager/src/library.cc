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
#include "tester.h"

namespace region_manager {

static const char* const library_name = "region_manager";

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);
  TesterTask::register_variants(context);
}

}  // namespace region_manager

extern "C" {

void perform_registration(void)
{
  legate::Core::perform_registration<region_manager::registration_callback>();
}
}
