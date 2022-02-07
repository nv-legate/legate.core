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

#ifndef LEGATE_USE_GASNET
#ifdef REALM_USE_GASNET1
#define LEGATE_USE_GASNET
#endif
#endif
