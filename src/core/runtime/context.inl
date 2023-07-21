/* Copyright 2023 NVIDIA Corporation
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

#include "core/runtime/context.h"

namespace legate {

#ifndef REALM_COMPILER_IS_NVCC

#ifdef LEGATE_USE_CUDA
extern Legion::Logger log_legate;
#endif

namespace detail {

template <typename REDOP>
void register_reduction_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto legion_redop_id = *static_cast<const int32_t*>(args.buffer.get_ptr());
  Legion::Runtime::register_reduction_op<REDOP>(legion_redop_id);
}

}  // namespace detail

template <typename REDOP>
int32_t LibraryContext::register_reduction_operator(int32_t redop_id)
{
  int32_t legion_redop_id = get_reduction_op_id(redop_id);
#ifdef LEGATE_USE_CUDA
  log_legate.error("Reduction operators must be registered in a .cu file when CUDA is enabled");
  LEGATE_ABORT;
#endif
  Legion::Runtime::perform_registration_callback(
    detail::register_reduction_callback<REDOP>,
    Legion::UntypedBuffer(&legion_redop_id, sizeof(int32_t)),
    true /*global*/);

  return legion_redop_id;
}

#else  // ifndef REALM_COMPILER_IS_NVCC

namespace detail {

template <typename T>
class CUDAReductionOpWrapper : public T {
 public:
  static const bool has_cuda_reductions = true;

  template <bool EXCLUSIVE>
  __device__ static void apply_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template apply<EXCLUSIVE>(lhs, rhs);
  }

  template <bool EXCLUSIVE>
  __device__ static void fold_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template fold<EXCLUSIVE>(lhs, rhs);
  }
};

template <typename REDOP>
void register_reduction_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto legion_redop_id = *static_cast<const int32_t*>(args.buffer.get_ptr());
  Legion::Runtime::register_reduction_op(
    legion_redop_id,
    Realm::ReductionOpUntyped::create_reduction_op<detail::CUDAReductionOpWrapper<REDOP>>(),
    nullptr,
    nullptr,
    false);
}

}  // namespace detail

template <typename REDOP>
int32_t LibraryContext::register_reduction_operator(int32_t redop_id)
{
  int32_t legion_redop_id = get_reduction_op_id(redop_id);
  Legion::Runtime::perform_registration_callback(
    detail::register_reduction_callback<REDOP>,
    Legion::UntypedBuffer(&legion_redop_id, sizeof(int32_t)),
    true /*global*/);
  return legion_redop_id;
}

#endif  // ifndef REALM_COMPILER_IS_NVCC

}  // namespace legate
