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
                     size_t ret)
    : Legion::TaskVariantRegistrar(tid, global, var_name),
      task_name(t_name),
      descriptor(desc),
      var(v),
      ret_size(ret)
  {
  }

  const char* task_name;
  Legion::CodeDescriptor descriptor;
  LegateVariantCode var;
  size_t ret_size;
};

void TaskRegistrar::record_variant(Legion::TaskID tid,
                                   const char* task_name,
                                   const Legion::CodeDescriptor& desc,
                                   Legion::ExecutionConstraintSet& execution_constraints,
                                   Legion::TaskLayoutConstraintSet& layout_constraints,
                                   LegateVariantCode var,
                                   Processor::Kind kind,
                                   const VariantOptions& options)
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
                                          options.return_size);

  registrar->execution_constraints.swap(execution_constraints);
  registrar->layout_constraints.swap(layout_constraints);
  registrar->add_constraint(Legion::ProcessorConstraint(kind));
  registrar->set_leaf(options.leaf);
  registrar->set_inner(options.inner);
  registrar->set_idempotent(options.idempotent);
  registrar->set_concurrent(options.concurrent);

  pending_task_variants_.push_back(registrar);
}

void TaskRegistrar::register_all_tasks(const LibraryContext& context)
{
  auto runtime = Legion::Runtime::get_runtime();
  // Do all our registrations
  for (auto& task : pending_task_variants_) {
    task->task_id =
      context.get_task_id(task->task_id);  // Convert a task local task id to a global id
    // Attach the task name too for debugging
    runtime->attach_name(task->task_id, task->task_name, false /*mutable*/, true /*local only*/);
    runtime->register_task_variant(*task, task->descriptor, nullptr, 0, task->ret_size, task->var);
    delete task;
  }
  pending_task_variants_.clear();
}

}  // namespace legate
