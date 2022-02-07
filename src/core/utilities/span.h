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

#include "legion.h"

namespace legate {

template <typename T>
struct Span {
 public:
  Span()            = default;
  Span(const Span&) = default;

 public:
  Span(T* data, size_t size) : data_(data), size_(size) {}

 public:
  size_t size() const { return size_; }

 public:
  decltype(auto) operator[](size_t pos)
  {
    assert(pos < size_);
    return data_[pos];
  }

 public:
  decltype(auto) subspan(size_t off)
  {
    assert(off <= size_);
    return Span(data_ + off, size_ - off);
  }

 public:
  const T* ptr() const { return data_; }

 private:
  T* data_{nullptr};
  size_t size_{0};
};

}  // namespace legate
