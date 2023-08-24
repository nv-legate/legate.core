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

#ifdef __cplusplus
// C++98 ignored?
#if __cplusplus <= 201103L
#define LEGATE_CPP_VERSION 11
#elif __cplusplus <= 201402L
#define LEGATE_CPP_VERSION 14
#elif __cplusplus <= 201703L
#define LEGATE_CPP_VERSION 17
#elif __cplusplus <= 202002L
#define LEGATE_CPP_VERSION 20
#else
#define LEGATE_CPP_VERSION 23  // current year, or date of c++2b ratification
#endif
#endif  // __cplusplus

#ifndef LEGATE_CPP_VERSION
#define LEGATE_CPP_VERSION 0
#endif

#define LEGATE_ABORT                                                                        \
  do {                                                                                      \
    legate::log_legate.error(                                                               \
      "Legate called abort in %s at line %d in function %s", __FILE__, __LINE__, __func__); \
    abort();                                                                                \
  } while (false)

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

#ifndef LEGATE_USE_NETWORK
#if defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_MPI) || \
  defined(REALM_USE_UCX)
#define LEGATE_USE_NETWORK
#endif
#endif

#ifdef LEGION_BOUNDS_CHECKS
#define LEGATE_BOUNDS_CHECKS
#endif

#define LEGATE_MAX_DIM LEGION_MAX_DIM

// TODO: 2022-10-04: Work around a Legion bug, by not instantiating futures on framebuffer.
#define LEGATE_NO_FUTURES_ON_FB
