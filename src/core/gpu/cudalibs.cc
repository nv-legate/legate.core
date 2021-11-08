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

#include "core/gpu/cudalibs.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define LEGATE_ABORT                                               \
  {                                                                \
    fprintf(stderr,                                                \
            "Legate called abort in %s at line %d in function %s", \
            __FILE__,                                              \
            __LINE__,                                              \
            __func__);                                             \
    abort();                                                       \
  }

namespace legate {

CUDALibraries::CUDALibraries(void) : cublas(NULL) {}

CUDALibraries::~CUDALibraries(void) { finalize(); }

void CUDALibraries::finalize(void)
{
  if (cublas != NULL) {
    cublasStatus_t status = cublasDestroy(cublas);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,
              "Internal Legate CUBLAS destruction failure "
              "with error code %d\n",
              status);
      LEGATE_ABORT
    }
    cublas = NULL;
  }
}

cublasContext* CUDALibraries::get_cublas(void)
{
  if (cublas == NULL) {
    cublasStatus_t status = cublasCreate(&cublas);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,
              "Internal Legate CUBLAS initialization failure "
              "with error code %d\n",
              status);
      LEGATE_ABORT
    }
    const char* disable_tensor_cores = getenv("LEGATE_DISABLE_TENSOR_CORES");
    if (disable_tensor_cores == NULL) {
      // No request to disable tensor cores so turn them on
      status = cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "WARNING: CUBLAS does not support Tensor cores!");
    }
  }
  return cublas;
}

}  // namespace legate
