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

#include "core/runtime/context.h"
#include "core/runtime/runtime.h"
#include "core/task/return.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/typedefs.h"

namespace legate {

// We're going to allow for each task to use only up to 170 scalar output stores
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 2048;

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
  static void register_variants();
  template <typename RET_T, typename REDUC_T>
  static void register_variants_with_return();

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
  static void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime)
  {
    if (!Core::show_progress) return;
    const auto exec_proc     = runtime->get_executing_processor(ctx);
    const auto proc_kind_str = (exec_proc.kind() == Legion::Processor::LOC_PROC)   ? "CPU"
                               : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU"
                                                                                   : "OpenMP";

    std::stringstream point_str;
    const auto& point = task->index_point;
    point_str << point[0];
    for (int32_t dim = 1; dim < task->index_point.dim; ++dim) point_str << "," << point[dim];

    log_legate.print("%s %s task, pt = (%s), proc = " IDFMT,
                     task_name(),
                     proc_kind_str,
                     point_str.str().c_str(),
                     exec_proc.id);
  }

  // Task wrappers so we can instrument all Legate tasks if we want
  template <LegateVariantImpl TASK_PTR>
  static ReturnValues legate_task_wrapper(const Legion::Task* task,
                                          const std::vector<Legion::PhysicalRegion>& regions,
                                          Legion::Context legion_context,
                                          Legion::Runtime* runtime)
  {
    show_progress(task, legion_context, runtime);

    TaskContext context(task, regions, legion_context, runtime);
    if (!Core::use_empty_task) (*TASK_PTR)(context);

    return context.pack_return_values();
  }

 public:
  // Methods for registering variants
  template <LegateVariantImpl TASK_PTR>
  static void register_variant(Legion::ExecutionConstraintSet& execution_constraints,
                               Legion::TaskLayoutConstraintSet& layout_constraints,
                               LegateVariantCode var,
                               Legion::Processor::Kind kind,
                               bool leaf       = false,
                               bool inner      = false,
                               bool idempotent = false)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    Legion::CodeDescriptor desc(
      Legion::LegionTaskWrapper::
        legion_task_wrapper<ReturnValues, LegateTask<T>::template legate_task_wrapper<TASK_PTR>>);
    auto task_id = T::TASK_ID;

    T::Registrar::record_variant(task_id,
                                 T::task_name(),
                                 desc,
                                 execution_constraints,
                                 layout_constraints,
                                 var,
                                 kind,
                                 leaf,
                                 inner,
                                 idempotent,
                                 LEGATE_MAX_SIZE_SCALAR_RETURN);
  }
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterCPUVariant {
 public:
  static void register_variant()
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::cpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_CPU_VARIANT,
                                                    Legion::Processor::LOC_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterCPUVariant<T, BASE, false> {
 public:
  static void register_variant()
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_OPENMP>
class RegisterOMPVariant {
 public:
  static void register_variant()
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::omp_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_OMP_VARIANT,
                                                    Legion::Processor::OMP_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterOMPVariant<T, BASE, false> {
 public:
  static void register_variant()
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_GPU>
class RegisterGPUVariant {
 public:
  static void register_variant()
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    BASE::template register_variant<T::gpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_GPU_VARIANT,
                                                    Legion::Processor::TOC_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterGPUVariant<T, BASE, false> {
 public:
  static void register_variant()
  {
    // Do nothing
  }
};

template <typename T>
/*static*/ void LegateTask<T>::register_variants()
{
  RegisterCPUVariant<T, LegateTask<T>, HasCPUVariant::value>::register_variant();
  RegisterOMPVariant<T, LegateTask<T>, HasOMPVariant::value>::register_variant();
  RegisterGPUVariant<T, LegateTask<T>, HasGPUVariant::value>::register_variant();
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
                      bool leaf,
                      bool inner,
                      bool idempotent,
                      size_t ret_size);

 public:
  void register_all_tasks(Legion::Runtime* runtime, LibraryContext& context);

 private:
  struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
   public:
    PendingTaskVariant(void)
      : Legion::TaskVariantRegistrar(), task_name(NULL), var(LEGATE_NO_VARIANT)
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
