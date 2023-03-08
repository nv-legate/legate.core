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

#include "core/data/store.h"
#include "core/utilities/typedefs.h"

#ifdef LEGATE_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <sstream>

/** @defgroup util Utilities
 */

/**
 * @file
 * @brief Debugging utilities
 */
namespace legate {

#ifdef LEGATE_USE_CUDA

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

inline bool is_device_only_ptr(const void* ptr)
{
  cudaPointerAttributes attrs;
  cudaError_t res = cudaPointerGetAttributes(&attrs, ptr);
  return res == cudaSuccess && attrs.type == cudaMemoryTypeDevice;
}

#endif  // LEGATE_USE_CUDA

/**
 * @ingroup util
 * @brief Converts the dense array into a string
 *
 * @param base Array to convert
 * @param extents Extents of the array
 * @param strides Strides for dimensions
 *
 * @return A string expressing the contents of the array
 */
template <typename T, int DIM>
std::string print_dense_array(const T* base, const Point<DIM>& extents, size_t strides[DIM])
{
#ifdef LEGATE_USE_CUDA
  T* buf = nullptr;
  if (is_device_only_ptr(base)) {
    size_t num_elems = 0;
    for (size_t dim = 0; dim < DIM; ++dim) {
      num_elems = MAX(num_elems, strides[dim] * extents[dim]);
    }
    buf             = new T[num_elems];
    cudaError_t res = cudaMemcpy(buf, base, num_elems * sizeof(T), cudaMemcpyDeviceToHost);
    assert(res == cudaSuccess);
    base = buf;
  }
#endif  // LEGATE_USE_CUDA
  std::stringstream ss;
  for (int dim = 0; dim < DIM; ++dim) {
    if (strides[dim] == 0) continue;
    ss << "[";
  }
  ss << *base;
  coord_t offset   = 0;
  Point<DIM> point = Point<DIM>::ZEROES();
  int dim;
  do {
    for (dim = DIM - 1; dim >= 0; --dim) {
      if (strides[dim] == 0) continue;
      if (point[dim] + 1 < extents[dim]) {
        ++point[dim];
        offset += strides[dim];
        ss << ", ";
        for (int i = dim + 1; i < DIM; ++i) {
          if (strides[i] == 0) continue;
          ss << "[";
        }
        ss << base[offset];
        break;
      } else {
        offset -= point[dim] * strides[dim];
        point[dim] = 0;
        ss << "]";
      }
    }
  } while (dim >= 0);
#ifdef LEGATE_USE_CUDA
  if (buf != nullptr) delete buf;
#endif  // LEGATE_USE_CUDA
  return ss.str();
}

/**
 * @ingroup util
 * @brief Converts the dense array into a string using an accessor
 *
 * @param accessor Accessor to an array
 * @param rect Sub-rectangle within which the elements should be retrieved
 *
 * @return A string expressing the contents of the array
 */
template <int DIM, typename ACC>
std::string print_dense_array(ACC accessor, const Rect<DIM>& rect)
{
  Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
  size_t strides[DIM];
  const typename ACC::value_type* base = accessor.ptr(rect, strides);
  return print_dense_array(base, extents, strides);
}

/**
 * @ingroup util
 * @brief Converts the store to a string
 *
 * @param store Store to convert
 *
 * @return A string expressing the contents of the store
 */
std::string print_dense_array(const Store& store);

}  // namespace legate
