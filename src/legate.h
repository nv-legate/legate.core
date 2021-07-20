/* Copyright 2021 NVIDIA Corporation
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

#ifndef __LEGATE_H__
#define __LEGATE_H__

#include <cxxabi.h>
#include <stdint.h>
#include <cstdlib>
#include <cstring>

#include "legion.h"
// legion.h has to go before this
#include "legate_c.h"

// Use this for type checking packing and unpacking of arguments
// between Python and C++, normally set to false except for debugging
#ifndef TYPE_SAFE_LEGATE
#define TYPE_SAFE_LEGATE false
#endif

#define LEGATE_ABORT                                                                        \
  {                                                                                         \
    legate::log_legate.error(                                                               \
      "Legate called abort in %s at line %d in function %s", __FILE__, __LINE__, __func__); \
    abort();                                                                                \
  }

#ifdef __CUDACC__
#define LEGATE_DEVICE_PREFIX __device__
#else
#define LEGATE_DEVICE_PREFIX
#endif

#ifndef LEGION_REDOP_HALF
#error "Legate needs Legion to be compiled with -DLEGION_REDOP_HALF"
#endif

#ifndef LEGATE_USE_CUDA
#ifdef LEGION_USE_CUDA
#define LEGATE_USE_CUDA
#endif
#endif

#ifndef LEGATE_USE_OPENMP
#ifdef REALM_USE_OPENMP
#define LEGATE_USE_OPENMP
#endif
#endif

#ifndef LEGATE_USE_GASNET
#ifdef REALM_USE_GASNET1
#define LEGATE_USE_GASNET
#endif
#endif

#if LEGION_MAX_DIM == 1

#define LEGATE_FOREACH_N(__func__) __func__(1)

#elif LEGION_MAX_DIM == 2

#define LEGATE_FOREACH_N(__func__) __func__(1) __func__(2)

#elif LEGION_MAX_DIM == 3

#define LEGATE_FOREACH_N(__func__) __func__(1) __func__(2) __func__(3)

#elif LEGION_MAX_DIM == 4

#define LEGATE_FOREACH_N(__func__) __func__(1) __func__(2) __func__(3) __func__(4)

#elif LEGION_MAX_DIM == 5

#define LEGATE_FOREACH_N(__func__) __func__(1) __func__(2) __func__(3) __func__(4) __func__(5)

#elif LEGION_MAX_DIM == 6

#define LEGATE_FOREACH_N(__func__) \
  __func__(1) __func__(2) __func__(3) __func__(4) __func__(5) __func__(6)

#elif LEGION_MAX_DIM == 7

#define LEGATE_FOREACH_N(__func__) \
  __func__(1) __func__(2) __func__(3) __func__(4) __func__(5) __func__(6) __func__(7)

#elif LEGION_MAX_DIM == 8

#define LEGATE_FOREACH_N(__func__) \
  __func__(1) __func__(2) __func__(3) __func__(4) __func__(5) __func__(6) __func__(7) __func__(8)

#elif LEGION_MAX_DIM == 9

#define LEGATE_FOREACH_N(__func__)                                                              \
  __func__(1) __func__(2) __func__(3) __func__(4) __func__(5) __func__(6) __func__(7) __func__( \
    8) __func__(9)

#else
#error "Unsupported LEGION_MAX_DIM"
#endif

#ifdef LEGATE_USE_CUDA
struct cublasContext;  // Use this as replacement for cublasHandle_t
#endif

namespace legate {

extern Legion::Logger log_legate;

template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename REDOP, bool EXCLUSIVE, int N, typename T = Legion::coord_t>
using AccessorRD = Legion::
  ReductionAccessor<REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorRO = Legion::FieldAccessor<READ_ONLY, FT, N, T>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorWO = Legion::FieldAccessor<WRITE_DISCARD, FT, N, T>;
template <typename FT, int N, typename T = Legion::coord_t>
using GenericAccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T>;

// C enum typedefs
typedef legate_core_variant_t LegateVariant;
typedef legate_core_partition_t LegatePartition;
typedef legate_core_type_code_t LegateTypeCode;
typedef legate_core_resource_t LegateResource;

// This maps a type to its LegateTypeCode
#if defined(__clang__) && !defined(__NVCC__)
template <class>
static constexpr LegateTypeCode legate_type_code_of = MAX_TYPE_NUMBER;

template <>
static constexpr LegateTypeCode legate_type_code_of<__half> = HALF_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<float> = FLOAT_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<double> = DOUBLE_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int8_t> = INT8_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int16_t> = INT16_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int32_t> = INT32_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int64_t> = INT64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint8_t> = UINT8_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint16_t> = UINT16_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint32_t> = UINT32_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint64_t> = UINT64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<bool> = BOOL_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<complex<float>> = COMPLEX64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<complex<double>> = COMPLEX128_LT;
#else  // not clang
template <class>
constexpr LegateTypeCode legate_type_code_of = MAX_TYPE_NUMBER;

template <>
constexpr LegateTypeCode legate_type_code_of<__half> = HALF_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<float> = FLOAT_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<double> = DOUBLE_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int8_t> = INT8_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int16_t> = INT16_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int32_t> = INT32_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int64_t> = INT64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint8_t> = UINT8_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint16_t> = UINT16_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint32_t> = UINT32_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint64_t> = UINT64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<bool> = BOOL_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<complex<float>> = COMPLEX64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<complex<double>> = COMPLEX128_LT;
#endif

template <typename T>
struct ReturnSize {
  static constexpr int32_t value = sizeof(T);
};

template <LegateTypeCode CODE>
struct LegateTypeOf {
  using type = void;
};
template <>
struct LegateTypeOf<LegateTypeCode::BOOL_LT> {
  using type = bool;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT8_LT> {
  using type = int8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT16_LT> {
  using type = int16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT32_LT> {
  using type = int32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT64_LT> {
  using type = int64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT8_LT> {
  using type = uint8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT16_LT> {
  using type = uint16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT32_LT> {
  using type = uint32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT64_LT> {
  using type = uint64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::HALF_LT> {
  using type = __half;
};
template <>
struct LegateTypeOf<LegateTypeCode::FLOAT_LT> {
  using type = float;
};
template <>
struct LegateTypeOf<LegateTypeCode::DOUBLE_LT> {
  using type = double;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX64_LT> {
  using type = complex<float>;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX128_LT> {
  using type = complex<double>;
};

template <LegateTypeCode CODE>
using legate_type_of = typename LegateTypeOf<CODE>::type;

template <LegateTypeCode CODE>
struct is_integral {
  static constexpr bool value = std::is_integral<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_signed {
  static constexpr bool value = std::is_signed<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_unsigned {
  static constexpr bool value = std::is_unsigned<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_floating_point {
  static constexpr bool value = std::is_floating_point<legate_type_of<CODE>>::value;
};

template <typename T>
struct is_complex : std::false_type {
};

template <>
struct is_complex<complex<float>> : std::true_type {
};

template <>
struct is_complex<complex<double>> : std::true_type {
};

struct LegateProjectionFunctor;

class Core {
 public:
  static void parse_config(void);
  static void shutdown(void);
  // Get layout constraints
  static Legion::LayoutConstraintID get_soa_layout(void);
  static Legion::LayoutConstraintID get_reduction_layout(Legion::ReductionOpID redop);
  static Legion::LayoutConstraintID get_virtual_layout(void);

 public:
  static LegateProjectionFunctor *get_projection_functor(Legion::ProjectionID functor_id);

 public:
  // Configuration settings
  static bool show_progress;

 public:
  static LegateTypeCode safe_cast_type_code(int code);
#ifdef LEGATE_USE_CUDA
 public:
  static cublasContext *get_cublas(void);
#endif
};

template <typename T>
class LegateTask {
 protected:
  // Helper class for checking for various kinds of variants
  struct HasCPUVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void cpu_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_cpu_variant(Check<void (Fallback::*)(void *), &U::cpu_variant> *);

    template <typename U>
    static yes &test_for_cpu_variant(...);

    static const bool value = (sizeof(test_for_cpu_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasSSEVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void sse_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_sse_variant(Check<void (Fallback::*)(void *), &U::sse_variant> *);

    template <typename U>
    static yes &test_for_sse_variant(...);

    static const bool value = (sizeof(test_for_sse_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasAVXVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void avx_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_avx_variant(Check<void (Fallback::*)(void *), &U::avx_variant> *);

    template <typename U>
    static yes &test_for_avx_variant(...);

    static const bool value = (sizeof(test_for_avx_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasOMPVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void omp_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_omp_variant(Check<void (Fallback::*)(void *), &U::omp_variant> *);

    template <typename U>
    static yes &test_for_omp_variant(...);

    static const bool value = (sizeof(test_for_omp_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasGPUVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void gpu_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_gpu_variant(Check<void (Fallback::*)(void *), &U::gpu_variant> *);

    template <typename U>
    static yes &test_for_gpu_variant(...);

    static const bool value = (sizeof(test_for_gpu_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasSUBCPUVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void sub_cpu_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_sub_cpu_variant(Check<void (Fallback::*)(void *), &U::sub_cpu_variant> *);

    template <typename U>
    static yes &test_for_sub_cpu_variant(...);

    static const bool value = (sizeof(test_for_sub_cpu_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasSUBGPUVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void sub_gpu_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_sub_gpu_variant(Check<void (Fallback::*)(void *), &U::sub_gpu_variant> *);

    template <typename U>
    static yes &test_for_sub_gpu_variant(...);

    static const bool value = (sizeof(test_for_sub_gpu_variant<Derived>(0)) == sizeof(yes));
  };
  struct HasSUBNUMAVariant {
    typedef char no[1];
    typedef char yes[2];

    struct Fallback {
      void sub_numa_variant(void *);
    };
    struct Derived : T, Fallback {
    };

    template <typename U, U>
    struct Check;

    template <typename U>
    static no &test_for_sub_numa_variant(Check<void (Fallback::*)(void *), &U::sub_numa_variant> *);

    template <typename U>
    static yes &test_for_sub_numa_variant(...);

    static const bool value = (sizeof(test_for_sub_numa_variant<Derived>(0)) == sizeof(yes));
  };

 public:
  static void register_variants(void);
  template <typename RET_T, typename REDUC_T>
  static void register_variants_with_return(void);
  template <typename TASK>
  static void set_layout_constraints(LegateVariant variant,
                                     Legion::TaskLayoutConstraintSet &layout_constraints);
  template <typename TASK>
  static void set_inner_constraints(LegateVariant variant,
                                    Legion::TaskLayoutConstraintSet &layout_constraints);

 public:
  static const char *task_name()
  {
    static std::string result;
    if (result.empty()) {
      int status      = 0;
      char *demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
      result          = demangled;
      free(demangled);
    }

    return result.c_str();
  }

  // Task wrappers so we can instrument all Legate tasks if we want
  template <void (*TASK_PTR)(const Legion::Task *,
                             const std::vector<Legion::PhysicalRegion> &,
                             Legion::Context,
                             Legion::Runtime *)>
  static void legate_task_wrapper(const Legion::Task *task,
                                  const std::vector<Legion::PhysicalRegion> &regions,
                                  Legion::Context ctx,
                                  Legion::Runtime *runtime)
  {
    if (Core::show_progress) {
      Legion::Processor exec_proc = runtime->get_executing_processor(ctx);
      switch (task->index_point.get_dim()) {
        case 1: {
          Legion::Point<1> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           exec_proc.id);
          break;
        }
#if LEGION_MAX_DIM > 1
        case 2: {
          Legion::Point<2> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 2
        case 3: {
          Legion::Point<3> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 3
        case 4: {
          Legion::Point<4> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 4
        case 5: {
          Legion::Point<5> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 5
        case 6: {
          Legion::Point<6> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           rank_point[5],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 6
        case 7: {
          Legion::Point<7> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT
                           "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           rank_point[5],
                           rank_point[6],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 7
        case 8: {
          Legion::Point<8> rank_point = task->index_point;
          log_legate.print(
            "%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
            task_name(),
            (exec_proc.kind() == Legion::Processor::LOC_PROC)
              ? "CPU"
              : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
            rank_point[0],
            rank_point[1],
            rank_point[2],
            rank_point[3],
            rank_point[4],
            rank_point[5],
            rank_point[6],
            rank_point[7],
            exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 8
        case 9: {
          Legion::Point<9> rank_point = task->index_point;
          log_legate.print(
            "%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
            task_name(),
            (exec_proc.kind() == Legion::Processor::LOC_PROC)
              ? "CPU"
              : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
            rank_point[0],
            rank_point[1],
            rank_point[2],
            rank_point[3],
            rank_point[4],
            rank_point[5],
            rank_point[6],
            rank_point[7],
            rank_point[8],
            exec_proc.id);
          break;
        }
#endif
        default: assert(false);
      }
    }
    (*TASK_PTR)(task, regions, ctx, runtime);
  }
  template <typename RET_T,
            RET_T (*TASK_PTR)(const Legion::Task *,
                              const std::vector<Legion::PhysicalRegion> &,
                              Legion::Context,
                              Legion::Runtime *)>
  static RET_T legate_task_wrapper(const Legion::Task *task,
                                   const std::vector<Legion::PhysicalRegion> &regions,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime)
  {
    if (Core::show_progress) {
      Legion::Processor exec_proc = runtime->get_executing_processor(ctx);
      switch (task->index_point.get_dim()) {
        case 1: {
          Legion::Point<1> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           exec_proc.id);
          break;
        }
#if LEGION_MAX_DIM > 1
        case 2: {
          Legion::Point<2> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 2
        case 3: {
          Legion::Point<3> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 3
        case 4: {
          Legion::Point<4> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 4
        case 5: {
          Legion::Point<5> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 5
        case 6: {
          Legion::Point<6> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           rank_point[5],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 6
        case 7: {
          Legion::Point<7> rank_point = task->index_point;
          log_legate.print("%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT
                           "\n",
                           task_name(),
                           (exec_proc.kind() == Legion::Processor::LOC_PROC)
                             ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
                           rank_point[0],
                           rank_point[1],
                           rank_point[2],
                           rank_point[3],
                           rank_point[4],
                           rank_point[5],
                           rank_point[6],
                           exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 7
        case 8: {
          Legion::Point<8> rank_point = task->index_point;
          log_legate.print(
            "%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
            task_name(),
            (exec_proc.kind() == Legion::Processor::LOC_PROC)
              ? "CPU"
              : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
            rank_point[0],
            rank_point[1],
            rank_point[2],
            rank_point[3],
            rank_point[4],
            rank_point[5],
            rank_point[6],
            rank_point[7],
            exec_proc.id);
          break;
        }
#endif
#if LEGION_MAX_DIM > 8
        case 9: {
          Legion::Point<9> rank_point = task->index_point;
          log_legate.print(
            "%s %s task, pt = (%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld), proc = " IDFMT "\n",
            task_name(),
            (exec_proc.kind() == Legion::Processor::LOC_PROC)
              ? "CPU"
              : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU" : "OpenMP",
            rank_point[0],
            rank_point[1],
            rank_point[2],
            rank_point[3],
            rank_point[4],
            rank_point[5],
            rank_point[6],
            rank_point[7],
            rank_point[8],
            exec_proc.id);
          break;
        }
#endif
        default: assert(false);
      }
    }
    return (*TASK_PTR)(task, regions, ctx, runtime);
  }

 public:
  // Methods for registering variants
  template <void (*TASK_PTR)(const Legion::Task *,
                             const std::vector<Legion::PhysicalRegion> &,
                             Legion::Context,
                             Legion::Runtime *)>
  static void register_variant(Legion::ExecutionConstraintSet &execution_constraints,
                               Legion::TaskLayoutConstraintSet &layout_constraints,
                               LegateVariant var,
                               Legion::Processor::Kind kind,
                               bool leaf       = false,
                               bool inner      = false,
                               bool idempotent = false)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    Legion::CodeDescriptor desc(Legion::LegionTaskWrapper::legion_task_wrapper<
                                LegateTask<T>::template legate_task_wrapper<TASK_PTR>>);
    T::record_variant(T::TASK_ID,
                      desc,
                      execution_constraints,
                      layout_constraints,
                      var,
                      kind,
                      leaf,
                      inner,
                      idempotent,
                      0 /*no return type*/);
  }
  template <typename RET_T,
            RET_T (*TASK_PTR)(const Legion::Task *,
                              const std::vector<Legion::PhysicalRegion> &,
                              Legion::Context,
                              Legion::Runtime *)>
  static void register_variant(Legion::ExecutionConstraintSet &execution_constraints,
                               Legion::TaskLayoutConstraintSet &layout_constraints,
                               LegateVariant var,
                               Legion::Processor::Kind kind,
                               bool leaf       = false,
                               bool inner      = false,
                               bool idempotent = false)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    Legion::CodeDescriptor desc(
      Legion::LegionTaskWrapper::
        legion_task_wrapper<RET_T, LegateTask<T>::template legate_task_wrapper<RET_T, TASK_PTR>>);
    T::record_variant(T::TASK_ID,
                      desc,
                      execution_constraints,
                      layout_constraints,
                      var,
                      kind,
                      leaf,
                      inner,
                      idempotent,
                      ReturnSize<RET_T>::value /*non void return type*/);
  }
};

template <typename T>
class SubLegateTask : public LegateTask<T> {
 public:
  static void sub_cpu_variant(const Legion::Task *task,
                              const std::vector<Legion::PhysicalRegion> &regions,
                              Legion::Context ctx,
                              Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_CPUS));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_CPUS);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    runtime->execute_index_space(ctx, launcher);
  }
  static void sub_gpu_variant(const Legion::Task *task,
                              const std::vector<Legion::PhysicalRegion> &regions,
                              Legion::Context ctx,
                              Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_GPUS));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_GPUS);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    runtime->execute_index_space(ctx, launcher);
  }
  static void sub_numa_variant(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx,
                               Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_NUMA));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_NUMA);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    runtime->execute_index_space(ctx, launcher);
  }
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterCPUVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_CPU_VARIANT, layout_constraints);
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
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterSSEVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    // This processor needs SSE instructions
    execution_constraints.add_constraint(
      Legion::ISAConstraint(LEGION_SSE_ISA | LEGION_SSE2_ISA | LEGION_SSE3_ISA | LEGION_SSE4_ISA));
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_SSE_VARIANT, layout_constraints);
    BASE::template register_variant<T::cpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_SSE_VARIANT,
                                                    Legion::Processor::LOC_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterSSEVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_CPU>
class RegisterAVXVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    // This processor needs SSE instructions and AVX instructions
    // No support for AVX2 yet
    execution_constraints.add_constraint(Legion::ISAConstraint(
      LEGION_SSE_ISA | LEGION_SSE2_ISA | LEGION_SSE3_ISA | LEGION_SSE4_ISA | LEGION_AVX_ISA));
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_AVX_VARIANT, layout_constraints);
    BASE::template register_variant<T::cpu_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_AVX_VARIANT,
                                                    Legion::Processor::LOC_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterAVXVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_OPENMP>
class RegisterOpenMPVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_OMP_VARIANT, layout_constraints);
    BASE::template register_variant<T::omp_variant>(execution_constraints,
                                                    layout_constraints,
                                                    LEGATE_OMP_VARIANT,
                                                    Legion::Processor::OMP_PROC,
                                                    true /*leaf*/);
  }
};

template <typename T, typename BASE>
class RegisterOpenMPVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_GPU>
class RegisterGPUVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_GPU_VARIANT, layout_constraints);
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
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_SUBRANK>
class RegisterSubrankCPUVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_CPU_VARIANT, layout_constraints);
    BASE::template register_variant<T::sub_cpu_variant>(execution_constraints,
                                                        layout_constraints,
                                                        LEGATE_SUB_CPU_VARIANT,
                                                        Legion::Processor::LOC_PROC,
                                                        false /*leaf*/,
                                                        true /*inner*/);
  }
};

template <typename T, typename BASE>
class RegisterSubrankCPUVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_SUBRANK>
class RegisterSubrankGPUVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_GPU_VARIANT, layout_constraints);
    BASE::template register_variant<T::sub_gpu_variant>(execution_constraints,
                                                        layout_constraints,
                                                        LEGATE_SUB_GPU_VARIANT,
                                                        Legion::Processor::LOC_PROC,
                                                        false /*leaf*/,
                                                        true /*inner*/);
  }
};

template <typename T, typename BASE>
class RegisterSubrankGPUVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, bool HAS_SUBRANK>
class RegisterSubrankNUMAVariant {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_OMP_VARIANT, layout_constraints);
    BASE::template register_variant<T::sub_numa_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_SUB_OMP_VARIANT,
                                                         Legion::Processor::LOC_PROC,
                                                         false /*leaf*/,
                                                         true /*inner*/);
  }
};

template <typename T, typename BASE>
class RegisterSubrankNUMAVariant<T, BASE, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T>
/*static*/ void LegateTask<T>::register_variants(void)
{
  RegisterCPUVariant<T, LegateTask<T>, HasCPUVariant::value>::register_variant();
  RegisterSSEVariant<T, LegateTask<T>, HasSSEVariant::value>::register_variant();
  RegisterAVXVariant<T, LegateTask<T>, HasAVXVariant::value>::register_variant();
  RegisterOpenMPVariant<T, LegateTask<T>, HasOMPVariant::value>::register_variant();
  RegisterGPUVariant<T, LegateTask<T>, HasGPUVariant::value>::register_variant();
  RegisterSubrankCPUVariant<T, LegateTask<T>, HasSUBCPUVariant::value>::register_variant();
  RegisterSubrankGPUVariant<T, LegateTask<T>, HasSUBGPUVariant::value>::register_variant();
  RegisterSubrankNUMAVariant<T, LegateTask<T>, HasSUBNUMAVariant::value>::register_variant();
}

// Another subrank class but for tasks with return values and reduction operations
template <typename T, typename RET_T>
class SubRetLegateTask : public LegateTask<T> {
 public:
  static RET_T sub_cpu_variant(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx,
                               Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_CPUS));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_CPUS);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    Legion::Future f = runtime->execute_index_space(ctx, launcher, T::REDOP);
    return f.get_result<RET_T>(true /*silence warnings*/);
  }
  static RET_T sub_gpu_variant(const Legion::Task *task,
                               const std::vector<Legion::PhysicalRegion> &regions,
                               Legion::Context ctx,
                               Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_GPUS));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_GPUS);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    Legion::Future f = runtime->execute_index_space(ctx, launcher, T::REDOP);
    return f.get_result<RET_T>(true /*silence warnings*/);
  }
  static RET_T sub_numa_variant(const Legion::Task *task,
                                const std::vector<Legion::PhysicalRegion> &regions,
                                Legion::Context ctx,
                                Legion::Runtime *runtime)
  {
    runtime->unmap_all_regions(ctx);
    // We must have at least one region requirement for this to work
    assert(!task->regions.empty());
    // Get the launch space
    Legion::IndexSpace launch_space = runtime->get_index_partition_color_space_name(
      runtime->get_index_partition(task->regions[0].region.get_index_space(), PID_NODE_NUMA));
    Legion::IndexTaskLauncher launcher(task->task_id,
                                       launch_space,
                                       Legion::TaskArgument(task->args, task->arglen),
                                       Legion::ArgumentMap());
    // Copy over the region requirements and everything else
    launcher.region_requirements = task->regions;
    // Convert the region requirements into projection requirements
    for (unsigned idx = 0; idx < regions.size(); idx++) {
      Legion::RegionRequirement &req = launcher.region_requirements[idx];
      req.parent                     = task->regions[idx].region;
      req.partition =
        runtime->get_logical_partition_by_color(task->regions[idx].region, PID_NODE_NUMA);
      req.handle_type = LEGION_PARTITION_PROJECTION;
      req.projection  = 0;  // identity projection function
    }
    // Also copy over any futures
    launcher.futures = task->futures;
    Legion::Future f = runtime->execute_index_space(ctx, launcher, T::REDOP);
    return f.get_result<RET_T>(true /*silence warnings*/);
  }
};

template <typename T, typename BASE, typename RET, bool HAS_CPU>
class RegisterCPUVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_CPU_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::cpu_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_CPU_VARIANT,
                                                         Legion::Processor::LOC_PROC,
                                                         true /*leaf*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterCPUVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_CPU>
class RegisterSSEVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    // This processor needs SSE instructions
    execution_constraints.add_constraint(
      Legion::ISAConstraint(LEGION_SSE_ISA | LEGION_SSE2_ISA | LEGION_SSE3_ISA | LEGION_SSE4_ISA));
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_SSE_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::cpu_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_SSE_VARIANT,
                                                         Legion::Processor::LOC_PROC,
                                                         true /*leaf*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterSSEVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_CPU>
class RegisterAVXVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    // This processor needs SSE and AVX instructions
    // No support needed for AVX2 yet
    execution_constraints.add_constraint(
      Legion::ISAConstraint(LEGION_SSE_ISA | LEGION_SSE2_ISA | LEGION_SSE3_ISA | LEGION_SSE4_ISA));
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_AVX_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::cpu_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_AVX_VARIANT,
                                                         Legion::Processor::LOC_PROC,
                                                         true /*leaf*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterAVXVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_OPENMP>
class RegisterOpenMPVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_OMP_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::omp_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_OMP_VARIANT,
                                                         Legion::Processor::OMP_PROC,
                                                         true /*leaf*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterOpenMPVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_GPU>
class RegisterGPUVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_layout_constraints<T>(LEGATE_GPU_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::gpu_variant>(execution_constraints,
                                                         layout_constraints,
                                                         LEGATE_GPU_VARIANT,
                                                         Legion::Processor::TOC_PROC,
                                                         true /*leaf*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterGPUVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_SUBRANK>
class RegisterSubrankCPUVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_CPU_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::sub_cpu_variant>(execution_constraints,
                                                             layout_constraints,
                                                             LEGATE_SUB_CPU_VARIANT,
                                                             Legion::Processor::LOC_PROC,
                                                             false /*leaf*/,
                                                             true /*inner*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterSubrankCPUVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_SUBRANK>
class RegisterSubrankGPUVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_GPU_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::sub_gpu_variant>(execution_constraints,
                                                             layout_constraints,
                                                             LEGATE_SUB_GPU_VARIANT,
                                                             Legion::Processor::LOC_PROC,
                                                             false /*leaf*/,
                                                             true /*inner*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterSubrankGPUVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T, typename BASE, typename RET, bool HAS_SUBRANK>
class RegisterSubrankNUMAVariantWithReturn {
 public:
  static void register_variant(void)
  {
    Legion::ExecutionConstraintSet execution_constraints;
    Legion::TaskLayoutConstraintSet layout_constraints;
    T::template set_inner_constraints<T>(LEGATE_SUB_OMP_VARIANT, layout_constraints);
    BASE::template register_variant<RET, T::sub_numa_variant>(execution_constraints,
                                                              layout_constraints,
                                                              LEGATE_SUB_OMP_VARIANT,
                                                              Legion::Processor::LOC_PROC,
                                                              false /*leaf*/,
                                                              true /*inner*/);
  }
};

template <typename T, typename BASE, typename RET>
class RegisterSubrankNUMAVariantWithReturn<T, BASE, RET, false> {
 public:
  static void register_variant(void)
  {
    // Do nothing
  }
};

template <typename T>
template <typename RET_T, typename REDUC_T>
/*static*/ void LegateTask<T>::register_variants_with_return(void)
{
  RegisterCPUVariantWithReturn<T, LegateTask<T>, RET_T, HasCPUVariant::value>::register_variant();
  RegisterSSEVariantWithReturn<T, LegateTask<T>, RET_T, HasSSEVariant::value>::register_variant();
  RegisterAVXVariantWithReturn<T, LegateTask<T>, RET_T, HasAVXVariant::value>::register_variant();
  RegisterOpenMPVariantWithReturn<T, LegateTask<T>, RET_T, HasOMPVariant::value>::
    register_variant();
  RegisterGPUVariantWithReturn<T, LegateTask<T>, REDUC_T, HasGPUVariant::value>::register_variant();
  RegisterSubrankCPUVariantWithReturn<T, LegateTask<T>, RET_T, HasSUBCPUVariant::value>::
    register_variant();
  RegisterSubrankGPUVariantWithReturn<T, LegateTask<T>, REDUC_T, HasSUBGPUVariant::value>::
    register_variant();
  RegisterSubrankNUMAVariantWithReturn<T, LegateTask<T>, RET_T, HasSUBNUMAVariant::value>::
    register_variant();
}

template <typename T>
template <typename TASK>
/*static*/ void LegateTask<T>::set_layout_constraints(
  LegateVariant variant, Legion::TaskLayoutConstraintSet &layout_constraints)
{
  // TODO: handle alignment constraints for different variant types
  for (int idx = 0; idx < TASK::REGIONS; idx++)
    layout_constraints.add_layout_constraint(idx, Core::get_soa_layout());
}

template <typename T>
template <typename TASK>
/*static*/ void LegateTask<T>::set_inner_constraints(
  LegateVariant variant, Legion::TaskLayoutConstraintSet &layout_constraints)
{
  for (int idx = 0; idx < TASK::REGIONS; idx++)
    layout_constraints.add_layout_constraint(idx, Core::get_virtual_layout());
}

// A class for helping with deserialization of arguments from python
class LegateDeserializer {
 public:
  LegateDeserializer(const void *a, size_t l) : args(static_cast<const char *>(a)), length(l) {}

 public:
  const void *data() const { return args; }
  void skip(size_t bytes) { args = args + bytes; }

 public:
  inline void check_type(int type_val, size_t type_size)
  {
    assert(length >= sizeof(int));
    int expected_type = *((const int *)args);
    length -= sizeof(int);
    args += sizeof(int);
    // The expected_type code is hardcoded in legate/core/legion.py
    assert(expected_type == type_val);
    assert(length >= type_size);
  }

 public:
  inline int16_t unpack_8bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT8, sizeof(int8_t));
    int8_t result = *((const int8_t *)args);
    length -= sizeof(int8_t);
    args += sizeof(int8_t);
    return result;
  }
  inline int16_t unpack_16bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT16, sizeof(int16_t));
    int16_t result = *((const int16_t *)args);
    length -= sizeof(int16_t);
    args += sizeof(int16_t);
    return result;
  }
  inline int32_t unpack_32bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT32, sizeof(int32_t));
    int32_t result = *((const int32_t *)args);
    length -= sizeof(int32_t);
    args += sizeof(int32_t);
    return result;
  }
  inline int64_t unpack_64bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT64, sizeof(int64_t));
    int64_t result = *((const int64_t *)args);
    length -= sizeof(int64_t);
    args += sizeof(int64_t);
    return result;
  }
  inline uint16_t unpack_8bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT8, sizeof(uint8_t));
    uint8_t result = *((const uint8_t *)args);
    length -= sizeof(uint8_t);
    args += sizeof(uint8_t);
    return result;
  }
  inline uint16_t unpack_16bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT16, sizeof(uint16_t));
    uint16_t result = *((const uint16_t *)args);
    length -= sizeof(uint16_t);
    args += sizeof(uint16_t);
    return result;
  }
  inline uint32_t unpack_32bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT32, sizeof(uint32_t));
    uint32_t result = *((const uint32_t *)args);
    length -= sizeof(uint32_t);
    args += sizeof(uint32_t);
    return result;
  }
  inline uint64_t unpack_64bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT64, sizeof(uint64_t));
    uint64_t result = *((const uint64_t *)args);
    length -= sizeof(uint64_t);
    args += sizeof(uint64_t);
    return result;
  }
  inline float unpack_float(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT32, sizeof(float));
    float result = *((const float *)args);
    length += sizeof(float);
    args += sizeof(float);
    return result;
  }
  inline double unpack_double(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT64, sizeof(double));
    double result = *((const double *)args);
    length -= sizeof(double);
    args += sizeof(double);
    return result;
  }
  inline bool unpack_bool(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_BOOL, sizeof(bool));
    bool result = *((const bool *)args);
    length -= sizeof(bool);
    args += sizeof(bool);
    return result;
  }
  inline __half unpack_half(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT16, sizeof(__half));
    __half result = *((const __half *)args);
    length -= sizeof(__half);
    args += sizeof(__half);
    return result;
  }
  inline char unpack_char(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_TOTAL + 1, sizeof(char));
    char result = *((const char *)args);
    length -= sizeof(char);
    args += sizeof(char);
    return result;
  }
  inline complex<float> unpack_64bit_complex(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_COMPLEX64, sizeof(complex<float>));
    complex<float> result = *((const complex<float> *)args);
    length -= sizeof(complex<float>);
    args += sizeof(complex<float>);
    return result;
  }
  inline complex<double> unpack_128bit_complex(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_COMPLEX128, sizeof(complex<double>));
    complex<double> result = *((const complex<double> *)args);
    length -= sizeof(complex<double>);
    args += sizeof(complex<double>);
    return result;
  }
  inline void *unpack_buffer(size_t buffer_size)
  {
    void *result = (void *)args;
    length -= buffer_size;
    args += buffer_size;
    return result;
  }

 public:
  template <typename T>
  inline T unpack_value(void)
  {
    assert(false);  // should never be called directly
    return T();
  }

 public:
  inline int unpack_dimension(void) { return unpack_32bit_int(); }
  template <int DIM>
  inline Legion::Point<DIM> unpack_point(void)
  {
    if (TYPE_SAFE_LEGATE) assert(unpack_32bit_int() == DIM);
    Legion::Point<DIM> result;
    for (int i = 0; i < DIM; i++) result[i] = unpack_64bit_int();
    return result;
  }
  template <int DIM>
  inline Legion::Rect<DIM> unpack_rect(void)
  {
    if (TYPE_SAFE_LEGATE) assert(unpack_32bit_int() == DIM);
    Legion::Rect<DIM> result;
    for (int i = 0; i < DIM; i++) result.lo[i] = unpack_64bit_int();
    for (int i = 0; i < DIM; i++) result.hi[i] = unpack_64bit_int();
    return result;
  }
  template <int M, int N>
  inline Legion::AffineTransform<M, N> unpack_transform(void)
  {
    Legion::AffineTransform<M, N> result;
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) result.transform[i][j] = unpack_64bit_int();
    for (int i = 0; i < M; i++) result.offset[i] = unpack_64bit_int();
    return result;
  }
  inline std::string unpack_string(void)
  {
    int size = unpack_32bit_int();
    std::string result;
    for (int i = 0; i < size; i++) result.push_back(unpack_char());
    return result;
  };
  inline LegateTypeCode unpack_dtype(void) { return Core::safe_cast_type_code(unpack_32bit_int()); }
  template <typename T, int DIM>
  inline AccessorRO<T, DIM> unpack_accessor_RO(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRO<T, DIM>(region, fid);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorRO<T, DIM>(region, fid, transform);                        \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRO<T, DIM>(region, fid);
  }
  template <typename T, int DIM>
  inline AccessorRO<T, DIM> unpack_accessor_RO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRO<T, DIM>(region, fid, shape);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorRO<T, DIM>(region, fid, transform, shape);                 \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRO<T, DIM>(region, fid, shape);
  }
  template <typename T, int DIM>
  inline AccessorRO<T, DIM> unpack_accessor_RO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RO<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    assert(extra_dim < (DIM + 1));
    Legion::AffineTransform<DIM + 1, DIM> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < DIM; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < (DIM + 1); d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    extra_transform.offset[extra_dim] = value;
    return AccessorRO<T, DIM>(region, fid, extra_transform, shape);
  }
  template <typename T, int DIM, int D2>
  inline AccessorRO<T, DIM> unpack_accessor_RO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RO<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == D2);
    Legion::AffineTransform<DIM, D2> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < D2; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < DIM; d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    assert(extra_dim < DIM);
    extra_transform.offset[extra_dim] = value;
    const int N                       = unpack_32bit_int();
    assert(N == DIM);
    Legion::AffineTransform<D2, DIM> transform = unpack_transform<D2, DIM>();
    return AccessorRO<T, DIM>(region, fid, extra_transform(transform), shape);
  }
  template <typename T, int DIM>
  inline AccessorWO<T, DIM> unpack_accessor_WO(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorWO<T, DIM>(region, fid);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorWO<T, DIM>(region, fid, transform);                        \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorWO<T, DIM>(region, fid);
  }
  template <typename T, int DIM>
  inline AccessorWO<T, DIM> unpack_accessor_WO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorWO<T, DIM>(region, fid, shape);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorWO<T, DIM>(region, fid, transform, shape);                 \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorWO<T, DIM>(region, fid, shape);
  }
  template <typename T, int DIM>
  inline AccessorWO<T, DIM> unpack_accessor_WO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_WO<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    assert(extra_dim < (DIM + 1));
    Legion::AffineTransform<DIM + 1, DIM> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < DIM; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < (DIM + 1); d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    extra_transform.offset[extra_dim] = value;
    return AccessorWO<T, DIM>(region, fid, extra_transform, shape);
  }
  template <typename T, int DIM, int D2>
  inline AccessorWO<T, DIM> unpack_accessor_WO(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_WO<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == D2);
    Legion::AffineTransform<DIM, D2> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < D2; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < DIM; d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    assert(extra_dim < DIM);
    extra_transform.offset[extra_dim] = value;
    const int N                       = unpack_32bit_int();
    assert(N == DIM);
    Legion::AffineTransform<D2, DIM> transform = unpack_transform<D2, DIM>();
    return AccessorWO<T, DIM>(region, fid, extra_transform(transform), shape);
  }
  template <typename T, int DIM>
  inline AccessorRW<T, DIM> unpack_accessor_RW(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRW<T, DIM>(region, fid);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorRW<T, DIM>(region, fid, transform);                        \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRW<T, DIM>(region, fid);
  }
  template <typename T, int DIM>
  inline AccessorRW<T, DIM> unpack_accessor_RW(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRW<T, DIM>(region, fid, shape);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                    \
  case DN: {                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>(); \
    return AccessorRW<T, DIM>(region, fid, transform, shape);                 \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRW<T, DIM>(region, fid, shape);
  }
  template <typename T, int DIM>
  inline AccessorRW<T, DIM> unpack_accessor_RW(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RW<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    assert(extra_dim < (DIM + 1));
    Legion::AffineTransform<DIM + 1, DIM> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < DIM; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < (DIM + 1); d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    extra_transform.offset[extra_dim] = value;
    return AccessorRW<T, DIM>(region, fid, extra_transform, shape);
  }
  template <typename T, int DIM, int D2>
  inline AccessorRW<T, DIM> unpack_accessor_RW(const Legion::PhysicalRegion &region,
                                               const Legion::Rect<DIM> &shape,
                                               const int extra_dim,
                                               const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RW<T, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == D2);
    Legion::AffineTransform<DIM, D2> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < D2; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < DIM; d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    assert(extra_dim < DIM);
    extra_transform.offset[extra_dim] = value;
    const int N                       = unpack_32bit_int();
    assert(N == DIM);
    Legion::AffineTransform<D2, DIM> transform = unpack_transform<D2, DIM>();
    return AccessorRW<T, DIM>(region, fid, extra_transform(transform), shape);
  }
  template <typename REDOP, bool EXCLUSIVE, int DIM>
  inline AccessorRD<typename REDOP::RHS, EXCLUSIVE, DIM> unpack_accessor_RD(
    const Legion::PhysicalRegion region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                             \
  case DN: {                                                                           \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>();          \
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID, transform); \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID);
  }
  template <typename REDOP, bool EXCLUSIVE, int DIM>
  inline AccessorRD<REDOP, EXCLUSIVE, DIM> unpack_accessor_RD(const Legion::PhysicalRegion region,
                                                              const Legion::Rect<DIM> &shape)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    if (M == 0) return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID, shape);
    const int N = unpack_32bit_int();
    assert(N == DIM);
    switch (M) {
#define LEGATE_DIMFUNC(DN)                                                                    \
  case DN: {                                                                                  \
    Legion::AffineTransform<DN, DIM> transform = unpack_transform<DN, DIM>();                 \
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID, transform, shape); \
  }
      LEGATE_FOREACH_N(LEGATE_DIMFUNC)
#undef LEGATE_DIMFUNC
      default: assert(false);
    }
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID, shape);
  }
  template <typename REDOP, bool EXCLUSIVE, int DIM>
  inline AccessorRD<REDOP, EXCLUSIVE, DIM> unpack_accessor_RD(const Legion::PhysicalRegion region,
                                                              const Legion::Rect<DIM> &shape,
                                                              const int extra_dim,
                                                              const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RD<REDOP, EXCLUSIVE, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    assert(extra_dim < (DIM + 1));
    Legion::AffineTransform<DIM + 1, DIM> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < DIM; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < (DIM + 1); d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    extra_transform.offset[extra_dim] = value;
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(region, fid, REDOP::REDOP_ID, extra_transform, shape);
  }
  template <typename REDOP, bool EXCLUSIVE, int DIM, int D2>
  inline AccessorRD<REDOP, EXCLUSIVE, DIM> unpack_accessor_RD(const Legion::PhysicalRegion region,
                                                              const Legion::Rect<DIM> &shape,
                                                              const int extra_dim,
                                                              const Legion::coord_t value)
  {
    if (extra_dim < 0) return unpack_accessor_RD<REDOP, EXCLUSIVE, DIM>(region, shape);
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == D2);
    Legion::AffineTransform<DIM, D2> extra_transform;
    // Zero out the diagonal
    for (int d = 0; d < D2; d++) extra_transform.transform[d][d] = 0;
    // Then fill in 1s on the dimensions that are not the extra one
    unsigned d2 = 0;
    for (int d1 = 0; d1 < DIM; d1++)
      if (d1 != extra_dim) extra_transform.transform[d1][d2++] = 1;
    assert(extra_dim < DIM);
    extra_transform.offset[extra_dim] = value;
    const int N                       = unpack_32bit_int();
    assert(N == DIM);
    Legion::AffineTransform<D2, DIM> transform = unpack_transform<D2, DIM>();
    return AccessorRD<REDOP, EXCLUSIVE, DIM>(
      region, fid, REDOP::REDOP_ID, extra_transform(transform), shape);
  }
  template <typename T, int DIM>
  inline GenericAccessorRO<T, DIM> unpack_generic_accessor_RO(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    return GenericAccessorRO<T, DIM>(region, fid);
  }
  template <typename T, int DIM>
  inline GenericAccessorWO<T, DIM> unpack_generic_accessor_WO(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    return GenericAccessorWO<T, DIM>(region, fid);
  }
  template <typename T, int DIM>
  inline GenericAccessorRW<T, DIM> unpack_generic_accessor_RW(const Legion::PhysicalRegion &region)
  {
    const Legion::FieldID fid = unpack_32bit_int();
    const int M               = unpack_32bit_int();
    assert(M == 0);
    return GenericAccessorRW<T, DIM>(region, fid);
  }

 protected:
  const char *args;
  size_t length;
};

template <>
inline int16_t LegateDeserializer::unpack_value<int16_t>(void)
{
  return unpack_16bit_int();
}
template <>
inline int32_t LegateDeserializer::unpack_value<int32_t>(void)
{
  return unpack_32bit_int();
}
template <>
inline int64_t LegateDeserializer::unpack_value<int64_t>(void)
{
  return unpack_64bit_int();
}
template <>
inline uint16_t LegateDeserializer::unpack_value<uint16_t>(void)
{
  return unpack_16bit_uint();
}
template <>
inline uint32_t LegateDeserializer::unpack_value<uint32_t>(void)
{
  return unpack_32bit_uint();
}
template <>
inline uint64_t LegateDeserializer::unpack_value<uint64_t>(void)
{
  return unpack_64bit_uint();
}
template <>
inline float LegateDeserializer::unpack_value<float>(void)
{
  return unpack_float();
}
template <>
inline double LegateDeserializer::unpack_value<double>(void)
{
  return unpack_double();
}
template <>
inline bool LegateDeserializer::unpack_value<bool>(void)
{
  return unpack_bool();
}
template <>
inline __half LegateDeserializer::unpack_value<__half>(void)
{
  return unpack_half();
}
template <>
inline complex<float> LegateDeserializer::unpack_value<complex<float>>(void)
{
  return unpack_64bit_complex();
}
template <>
inline complex<double> LegateDeserializer::unpack_value<complex<double>>(void)
{
  return unpack_128bit_complex();
}

}  // namespace legate

#endif  // __LEGATE_H__
