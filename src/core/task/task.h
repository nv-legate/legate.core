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

#include <cxxabi.h>
#include <sstream>

#include "legion.h"
#include "realm/faults.h"

#include "core/runtime/context.h"
#include "core/runtime/runtime.h"
#include "core/task/exception.h"
#include "core/task/return.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

namespace legate {

// We're going to allow for each task to use only up to 341 scalar output stores
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

struct VariantOptions {
  bool leaf{true};
  bool inner{false};
  bool idempotent{false};
  bool concurrent{false};
  size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  VariantOptions& with_leaf(bool _leaf)
  {
    leaf = _leaf;
    return *this;
  }
  VariantOptions& with_inner(bool _inner)
  {
    inner = _inner;
    return *this;
  }
  VariantOptions& with_idempotent(bool _idempotent)
  {
    idempotent = _idempotent;
    return *this;
  }
  VariantOptions& with_concurrent(bool _concurrent)
  {
    concurrent = _concurrent;
    return *this;
  }
  VariantOptions& with_return_size(size_t _return_size)
  {
    return_size = _return_size;
    return *this;
  }
};

using LegateVariantImpl = void (*)(TaskContext&);

template <typename T>
class LegateTask {
 protected:
  // Helper class for checking for various kinds of variants
  using __no  = int8_t[1];
  using __yes = int8_t[2];
  struct HasCPUVariant {
    template <typename U>
    static __yes& test(decltype(&U::cpu_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };
  struct HasOMPVariant {
    template <typename U>
    static __yes& test(decltype(&U::omp_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };
  struct HasGPUVariant {
    template <typename U>
    static __yes& test(decltype(&U::gpu_variant));
    template <typename U>
    static __no& test(...);
    static const bool value = (sizeof(test<T>(0)) == sizeof(__yes));
  };

 public:
  static const char* task_name()
  {
    static std::string result;
    if (result.empty()) {
      int status      = 0;
      char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
      result          = demangled;
      free(demangled);
    }

    return result.c_str();
  }

  // Task wrappers so we can instrument all Legate tasks if we want
  template <LegateVariantImpl TASK_PTR>
  static void legate_task_wrapper(
    const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
  {
    // Legion preamble
    const Legion::Task* task;
    const std::vector<Legion::PhysicalRegion>* regions;
    Legion::Context legion_context;
    Legion::Runtime* runtime;
    Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

#ifdef LEGATE_USE_CUDA
    nvtx::Range auto_range(task_name());
#endif

    Core::show_progress(task, legion_context, runtime, task_name());

    TaskContext context(task, *regions, legion_context, runtime);

    ReturnValues return_values{};
    try {
      if (!Core::use_empty_task) (*TASK_PTR)(context);
      return_values = context.pack_return_values();
    } catch (legate::TaskException& e) {
      if (context.can_raise_exception()) {
        context.make_all_unbound_stores_empty();
        return_values = context.pack_return_values_with_exception(e.index(), e.error_message());
      } else
        // If a Legate exception is thrown by a task that does not declare any exception,
        // this is a bug in the library that needs to be reported to the developer
        Core::report_unexpected_exception(task_name(), e);
    }

    // Legion postamble
    return_values.finalize(legion_context);
  }

 public:
  // Methods for registering variants
  template <LegateVariantImpl TASK_PTR>
  static void register_variant(Legion::ExecutionConstraintSet& execution_constraints,
                               Legion::TaskLayoutConstraintSet& layout_constraints,
                               LegateVariantCode var,
                               Legion::Processor::Kind kind,
                               const VariantOptions& options)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    Legion::CodeDescriptor desc(legate_task_wrapper<TASK_PTR>);
    auto task_id = T::TASK_ID;

    T::Registrar::record_variant(
      task_id, T::task_name(), desc, execution_constraints, layout_constraints, var, kind, options);
  }
  static void register_variants(
    const std::map<LegateVariantCode, VariantOptions>& all_options = {});
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterCPUVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::cpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_CPU_VARIANT,
                                                    Legion::Processor::LOC_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterCPUVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_OPENMP>
class RegisterOMPVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::omp_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_OMP_VARIANT,
                                                    Legion::Processor::OMP_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterOMPVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_GPU>
class RegisterGPUVariant {
 public:
  static void register_variant(const VariantOptions& options)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::gpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_GPU_VARIANT,
                                                    Legion::Processor::TOC_PROC,
                                                    options);
  }
};

template <typename T, typename BASE>
class RegisterGPUVariant<T, BASE, false> {
 public:
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  // Make a copy of the map of options so that we can do find-or-create on it
  auto all_options_copy = all_options;
  RegisterCPUVariant<T, LegateTask<T>, HasCPUVariant::value>::register_variant(
    all_options_copy[LEGATE_CPU_VARIANT]);
  RegisterOMPVariant<T, LegateTask<T>, HasOMPVariant::value>::register_variant(
    all_options_copy[LEGATE_OMP_VARIANT]);
  RegisterGPUVariant<T, LegateTask<T>, HasGPUVariant::value>::register_variant(
    all_options_copy[LEGATE_GPU_VARIANT]);
}

class LegateTaskRegistrar {
 public:
  void record_variant(Legion::TaskID tid,
                      const char* task_name,
                      const Legion::CodeDescriptor& desc,
                      Legion::ExecutionConstraintSet& execution_constraints,
                      Legion::TaskLayoutConstraintSet& layout_constraints,
                      LegateVariantCode var,
                      Legion::Processor::Kind kind,
                      const VariantOptions& options);

 public:
  void register_all_tasks(Legion::Runtime* runtime, LibraryContext& context);

 private:
  struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
   public:
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

   public:
    const char* task_name;
    Legion::CodeDescriptor descriptor;
    LegateVariantCode var;
    size_t ret_size;
  };

 private:
  std::vector<PendingTaskVariant> pending_task_variants_;
};

}  // namespace legate
