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
#include "produce_normal.h"
#include "produce_unbound.h"
#include "reduce_normal.h"
#include "reduce_unbound.h"

namespace tree_reduce {

static const char* const library_name = "tree_reduce";

Legion::Logger log_tree_reduce(library_name);

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  ProduceUnboundTask::register_variants(context);
  ReduceUnboundTask::register_variants(context);
  ProduceNormalTask::register_variants(context);
  ReduceNormalTask::register_variants(context);
}

}  // namespace tree_reduce

extern "C" {

void perform_registration(void)
{
  legate::Core::perform_registration<tree_reduce::registration_callback>();
}
}
