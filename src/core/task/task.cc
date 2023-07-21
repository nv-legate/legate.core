/* Copyright 2021-2022 NVIDIA Corporation
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

#include "core/task/task.h"

#include <cxxabi.h>

#include "realm/faults.h"

#include "core/runtime/context.h"
#include "core/runtime/runtime.h"
#include "core/task/exception.h"
#include "core/task/registrar.h"
#include "core/task/return.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

namespace legate {
namespace detail {

std::string generate_task_name(const std::type_info& ti)
{
  std::string result;
  int status      = 0;
  char* demangled = abi::__cxa_demangle(ti.name(), 0, 0, &status);
  result          = demangled;
  free(demangled);
  return std::move(result);
}

void task_wrapper(VariantImpl variant_impl,
                  const std::string& task_name,
                  const void* args,
                  size_t arglen,
                  const void* userdata,
                  size_t userlen,
                  Processor p)

{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

#ifdef LEGATE_USE_CUDA
  std::stringstream ss;
  ss << task_name;
  if (!task->get_provenance_string().empty()) ss << " : " + task->get_provenance_string();
  std::string msg = ss.str();
  nvtx::Range auto_range(msg.c_str());
#endif

  Core::show_progress(task, legion_context, runtime);

  TaskContext context(task, *regions, legion_context, runtime);

  ReturnValues return_values{};
  try {
    if (!Core::use_empty_task) (*variant_impl)(context);
    return_values = context.pack_return_values();
  } catch (legate::TaskException& e) {
    if (context.can_raise_exception()) {
      context.make_all_unbound_stores_empty();
      return_values = context.pack_return_values_with_exception(e.index(), e.error_message());
    } else
      // If a Legate exception is thrown by a task that does not declare any exception,
      // this is a bug in the library that needs to be reported to the developer
      Core::report_unexpected_exception(task, e);
  }

  // Legion postamble
  return_values.finalize(legion_context);
}

}  // namespace detail
}  // namespace legate
