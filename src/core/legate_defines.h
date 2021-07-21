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

#pragma once

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
