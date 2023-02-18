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

#pragma once

#include "core/task/variant.h"
#include "core/utilities/typedefs.h"

namespace legate {

class TaskContext;

using VariantImpl = void (*)(TaskContext&);

template <typename T>
struct LegateTask {
  // Exports the base class so we can access it via subclass T
  using BASE = LegateTask<T>;

  static void register_variants(
    const std::map<LegateVariantCode, VariantOptions>& all_options = {});

 private:
  template <typename, template <typename...> typename, bool>
  friend struct detail::RegisterVariantImpl;

  // A wrapper that wraps all Legate task variant implementations. Provides
  // common functionalities and instrumentations
  template <VariantImpl VARIANT_IMPL>
  static void legate_task_wrapper(
    const void* args, size_t arglen, const void* userdata, size_t userlen, Processor p);

  // A helper to register a single task variant
  template <VariantImpl VARIANT_IMPL>
  static void register_variant(Legion::ExecutionConstraintSet& execution_constraints,
                               Legion::TaskLayoutConstraintSet& layout_constraints,
                               LegateVariantCode var,
                               Processor::Kind kind,
                               const VariantOptions& options);

  static const char* task_name();
};

}  // namespace legate

#include "core/task/task.inl"
