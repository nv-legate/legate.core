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

/**
 * @file
 * @brief Class definition fo legate::LegateTask
 */
namespace legate {

// We're going to allow for each task to use only up to 341 scalar output stores
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

/**
 * @brief A helper class for specifying variant options
 */
struct VariantOptions {
  /**
   * @brief If the flag is `true`, the variant launches no subtasks. `true` by default.
   */
  bool leaf{true};
  bool inner{false};
  bool idempotent{false};
  /**
   * @brief If the flag is `true`, the variant needs a concurrent task launch. `false` by default.
   */
  bool concurrent{false};
  /**
   * @brief Maximum aggregate size for scalar output values. 4096 by default.
   */
  size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  /**
   * @brief Changes the value of the `leaf` flag
   *
   * @param `_leaf` A new value for the `leaf` flag
   */
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
  /**
   * @brief Changes the value of the `concurrent` flag
   *
   * @param `_concurrent` A new value for the `concurrent` flag
   */
  VariantOptions& with_concurrent(bool _concurrent)
  {
    concurrent = _concurrent;
    return *this;
  }
  /**
   * @brief Sets a maximum aggregate size for scalar output values
   *
   * @param `_return_size` A new maximum aggregate size for scalar output values
   */
  VariantOptions& with_return_size(size_t _return_size)
  {
    return_size = _return_size;
    return *this;
  }
};

/**
 * @brief A base class for Legate task implementations
 *
 * Any Legate task must inherit legate::LegateTask directly of transitively. Curently, each task
 * can have up to three variants and the variants need to be static member functions of the class
 * under the following names:
 *
 *   - `cpu_variant`: CPU implementation of the task
 *   - `gpu_variant`: GPU implementation of the task
 *   - `omp_variant`: OpenMP implementation of the task
 *
 * Tasks must have at least one variant, and all task variants must be semantically equivalent
 * (modulo some minor rounding errors due to floating point imprecision).
 */
template <typename T>
class LegateTask {
 public:
  /**
   * @brief Function signature for task variants. Each task variant must be a function of this type.
   */
  using LegateVariantImpl = void (*)(TaskContext&);

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
  /**
   * @brief Registers all task variants of the task. The client can optionally specifies
   * variant options.
   *
   * @all_options Options for task variants. Variants with no entires in `all_options` will use
   * the default set of options
   */
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
