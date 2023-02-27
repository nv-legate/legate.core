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

#include <memory>

#include "legion.h"

#include "core/task/variant.h"
#include "core/utilities/typedefs.h"

namespace legate {

class LibraryContext;
class PendingTaskVariant;

class TaskRegistrar {
 public:
  void record_variant(Legion::TaskID tid,
                      const char* task_name,
                      const Legion::CodeDescriptor& desc,
                      Legion::ExecutionConstraintSet& execution_constraints,
                      Legion::TaskLayoutConstraintSet& layout_constraints,
                      LegateVariantCode var,
                      Processor::Kind kind,
                      const VariantOptions& options);

 public:
  void register_all_tasks(const LibraryContext& context);

 private:
  std::vector<PendingTaskVariant*> pending_task_variants_;
};

}  // namespace legate
