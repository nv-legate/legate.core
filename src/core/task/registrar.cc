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

#include "core/task/registrar.h"

#include "core/runtime/context.h"
#include "core/task/task_info.h"
#include "core/utilities/typedefs.h"

namespace legate {

struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
  PendingTaskVariant(void)
    : Legion::TaskVariantRegistrar(), task_name(nullptr), var(LEGATE_NO_VARIANT)
  {
  }
  PendingTaskVariant(Legion::TaskID tid,
                     bool global,
                     const char* var_name,
                     const char* t_name,
                     const Legion::CodeDescriptor& desc,
                     LegateVariantCode v,
                     const VariantOptions& _options,
                     VariantImpl _body)
    : Legion::TaskVariantRegistrar(tid, global, var_name),
      task_name(t_name),
      descriptor(desc),
      var(v),
      options(_options),
      body(_body)
  {
  }

  const char* task_name;
  Legion::CodeDescriptor descriptor;
  LegateVariantCode var;
  VariantOptions options;
  VariantImpl body;
};

void TaskRegistrar::record_variant(Legion::TaskID tid,
                                   const char* task_name,
                                   const Legion::CodeDescriptor& desc,
                                   LegateVariantCode var,
                                   Processor::Kind kind,
                                   const VariantOptions& options,
                                   VariantImpl body)
{
  assert((kind == Processor::LOC_PROC) || (kind == Processor::TOC_PROC) ||
         (kind == Processor::OMP_PROC));

  // Buffer these up until we can do our actual registration with the runtime
  auto registrar = new PendingTaskVariant(tid,
                                          false /*global*/,
                                          (kind == Processor::LOC_PROC)   ? "CPU"
                                          : (kind == Processor::TOC_PROC) ? "GPU"
                                                                          : "OpenMP",
                                          task_name,
                                          desc,
                                          var,
                                          options,
                                          body);

  registrar->add_constraint(Legion::ProcessorConstraint(kind));
  registrar->set_leaf(options.leaf);
  registrar->set_inner(options.inner);
  registrar->set_idempotent(options.idempotent);
  registrar->set_concurrent(options.concurrent);

  pending_task_variants_.push_back(registrar);
}

void TaskRegistrar::register_all_tasks(LibraryContext& context)
{
  auto runtime = Legion::Runtime::get_runtime();
  std::unordered_map<int64_t, std::unique_ptr<TaskInfo>> task_infos;
  for (auto& task : pending_task_variants_) {
    TaskInfo* info;
    auto finder = task_infos.find(task->task_id);
    if (task_infos.end() == finder) {
      auto p_info = std::make_unique<TaskInfo>(task->task_name);
      info        = p_info.get();
      task_infos.emplace(std::make_pair(task->task_id, std::move(p_info)));
    } else
      info = finder->second.get();
    info->add_variant(task->var, task->body, task->options);
  }

  for (auto& [task_id, task_info] : task_infos) context.record_task(task_id, std::move(task_info));

  // Do all our registrations
  for (auto& task : pending_task_variants_) {
    task->task_id =
      context.get_task_id(task->task_id);  // Convert a task local task id to a global id
    // Attach the task name too for debugging
    runtime->attach_name(task->task_id, task->task_name, false /*mutable*/, true /*local only*/);
    runtime->register_task_variant(
      *task, task->descriptor, nullptr, 0, task->options.return_size, task->var);
    delete task;
  }
  pending_task_variants_.clear();
}

}  // namespace legate
