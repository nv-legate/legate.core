/* Copyright 2022 NVIDIA Corporation
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

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(expr)                                      \
  do {                                                        \
    cudaError_t __result__ = (expr);                          \
    legate::cuda::check_cuda(__result__, __FILE__, __LINE__); \
  } while (false)

#ifdef DEBUG_LEGATE

#define CHECK_CUDA_STREAM(stream)              \
  do {                                         \
    CHECK_CUDA(cudaStreamSynchronize(stream)); \
    CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)

#else

#define CHECK_CUDA_STREAM(stream)

#endif

namespace legate {
namespace cuda {

__host__ inline void check_cuda(cudaError_t error, const char* file, int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr,
            "Internal CUDA failure with error %s (%s) in file %s at line %d\n",
            cudaGetErrorString(error),
            cudaGetErrorName(error),
            file,
            line);
#ifdef DEBUG_LEGATE
    assert(false);
#else
    exit(error);
#endif
  }
}

}  // namespace cuda
}  // namespace legate
