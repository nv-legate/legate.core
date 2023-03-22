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

#include <assert.h>
#include <stddef.h>

/**
 * @file
 * @brief Class definition for legate::Span
 */

namespace legate {

/**
 * @ingroup data
 * @brief A simple span implementation used in Legate. Should eventually be replaced with
 * std::span once we bump up the C++ standard version to C++20
 */
template <typename T>
struct Span {
 public:
  Span()            = default;
  Span(const Span&) = default;

 public:
  /**
   * @brief Creates a span with an existing pointer and a size.
   *
   * The caller must guarantee that the allocation is big enough (i.e., bigger than or
   * equal to `sizeof(T) * size`) and that the allocation is alive while the span is alive.
   *
   * @param data Pointer to the data
   * @param size Number of elements
   */
  Span(T* data, size_t size) : data_(data), size_(size) {}

 public:
  /**
   * @brief Returns the number of elements
   *
   * @return The number of elements
   */
  size_t size() const { return size_; }

 public:
  decltype(auto) operator[](size_t pos) const
  {
    assert(pos < size_);
    return data_[pos];
  }
  /**
   * @brief Returns the pointer to the first element
   *
   * @return Pointer to the first element
   */
  const T* begin() const { return &data_[0]; }
  /**
   * @brief Returns the pointer to the end of allocation
   *
   * @return Pointer to the end of allocation
   */
  const T* end() const { return &data_[size_]; }

 public:
  /**
   * @brief Slices off the first `off` elements. Passing an `off` greater than
   * the size will fail with an assertion failure.
   *
   * @param off Number of elements to skip
   *
   * @return A span for range `[off, size())`
   */
  decltype(auto) subspan(size_t off)
  {
    assert(off <= size_);
    return Span(data_ + off, size_ - off);
  }

 public:
  /**
   * @brief Returns a `const` pointer to the data
   *
   * @return Pointer to the data
   */
  const T* ptr() const { return data_; }

 private:
  T* data_{nullptr};
  size_t size_{0};
};

}  // namespace legate
