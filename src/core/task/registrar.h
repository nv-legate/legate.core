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

/**
 * @file
 * @brief Class definition fo legate::TaskRegistrar
 */

namespace legate {

class LibraryContext;
class TaskInfo;

/**
 * @ingroup task
 * @brief A helper class for task variant registration.
 *
 * The `legate::TaskRegistrar` class is designed to simplify the boilerplate that client libraries
 * need to register all its task variants. The following is a boilerplate that each library
 * needs to write:
 *
 * @code{.cpp}
 * struct MyLibrary {
 *  public:
 *   template <typename... Args>
 *   static void record_variant(Args&&... args)
 *   {
 *     get_registrar().record_variant(std::forward<Args>(args)...);
 *   }
 *   static legate::TaskRegistrar& get_registrar();
 * };
 *
 * template <typename T>
 * struct MyLibraryTaskBase : public legate::LegateTask<T> {
 *   using Registrar = MyLibrary;
 *
 *   ...
 * };
 * @endcode
 *
 * In the code above, the `MyLibrary` has a static member that returns a singleton
 * `legate::TaskRegistrar` object, and another member `record_variant` that simply forwards all
 * arguments to the registrar. Then, the `MyLibraryTaskBase` points to the class so Legate can find
 * where task variants are registered.
 *
 * Once this registrar is set up in a library, each library task can simply register itself
 * with the `LegateTask::register_variants` method like the following:
 *
 * @code{.cpp}
 * // In a header
 * struct MyLibraryTask : public MyLibraryTaskBase<MyLibraryTask> {
 *   ...
 * };
 *
 * // In a C++ file
 * static void __attribute__((constructor)) register_tasks()
 * {
 *   MyLibraryTask::register_variants();
 * }
 * @endcode
 */
class TaskRegistrar {
 public:
  void record_variant(int64_t local_task_id,
                      LegateVariantCode variant_id,
                      const std::string& task_name,
                      VariantImpl body,
                      const Legion::CodeDescriptor& code_desc,
                      const VariantOptions& options);

 public:
  /**
   * @brief Registers all tasks recorded in this registrar. Typically invoked in a registration
   * callback of a library.
   *
   * @param context Context of the library that owns this registrar
   */
  void register_all_tasks(LibraryContext& context);

 private:
  std::unordered_map<int64_t, std::unique_ptr<TaskInfo>> pending_task_infos_;
};

}  // namespace legate
