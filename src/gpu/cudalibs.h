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

//#include <cublas_v2.h>
// We can't include cublas_v2.h because it sucks in half precision
// types into our CPU code which then conflict with legion's half
// precision types, so we mirror the cublas initialization code here
struct cublasContext;

namespace legate {

struct CUDALibraries {
 public:
  CUDALibraries(void);
  ~CUDALibraries(void);

 private:
  // Prevent copying and overwriting
  CUDALibraries(const CUDALibraries& rhs);
  CUDALibraries& operator=(const CUDALibraries& rhs);

 public:
  void finalize(void);
  cublasContext* get_cublas(void);

 protected:
  cublasContext* cublas;  // this is synonymous with cublasHandle_t
};

}  // namespace legate
