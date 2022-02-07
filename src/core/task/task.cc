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

namespace legate {

using namespace Legion;

void LegateTaskRegistrar::record_variant(TaskID tid,
                                         const char* task_name,
                                         const CodeDescriptor& descriptor,
                                         ExecutionConstraintSet& execution_constraints,
                                         TaskLayoutConstraintSet& layout_constraints,
                                         LegateVariantCode var,
                                         Processor::Kind kind,
                                         bool leaf,
                                         bool inner,
                                         bool idempotent,
                                         size_t ret_size)
{
  assert((kind == Processor::LOC_PROC) || (kind == Processor::TOC_PROC) ||
         (kind == Processor::OMP_PROC));

  // Buffer these up until we can do our actual registration with the runtime
  pending_task_variants_.push_back(PendingTaskVariant(tid,
                                                      false /*global*/,
                                                      (kind == Processor::LOC_PROC)   ? "CPU"
                                                      : (kind == Processor::TOC_PROC) ? "GPU"
                                                                                      : "OpenMP",
                                                      task_name,
                                                      descriptor,
                                                      var,
                                                      ret_size));

  auto& registrar = pending_task_variants_.back();
  registrar.execution_constraints.swap(execution_constraints);
  registrar.layout_constraints.swap(layout_constraints);
  registrar.add_constraint(ProcessorConstraint(kind));
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
}

void LegateTaskRegistrar::register_all_tasks(Runtime* runtime, LibraryContext& context)
{
  // Do all our registrations
  for (auto& task : pending_task_variants_) {
    task.task_id =
      context.get_task_id(task.task_id);  // Convert a task local task id to a global id
    // Attach the task name too for debugging
    runtime->attach_name(task.task_id, task.task_name, false /*mutable*/, true /*local only*/);
    runtime->register_task_variant(task, task.descriptor, NULL, 0, task.ret_size, task.var);
  }
  pending_task_variants_.clear();
}

}  // namespace legate
