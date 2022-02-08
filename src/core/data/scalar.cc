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

#include "core/data/scalar.h"
#include "core/utilities/dispatch.h"

namespace legate {

Scalar::Scalar(const Scalar& other) : own_(other.own_), tuple_(other.tuple_), code_(other.code_)
{
  copy(other);
}

Scalar::Scalar(bool tuple, LegateTypeCode code, const void* data)
  : tuple_(tuple), code_(code), data_(data)
{
}

Scalar::~Scalar()
{
  if (own_)
    // We know we own this buffer
    free(const_cast<void*>(data_));
}

Scalar& Scalar::operator=(const Scalar& other)
{
  own_   = other.own_;
  tuple_ = other.tuple_;
  code_  = other.code_;
  copy(other);
  return *this;
}

void Scalar::copy(const Scalar& other)
{
  if (other.own_) {
    auto size   = other.size();
    auto buffer = malloc(size);
    memcpy(buffer, other.data_, size);
    data_ = buffer;
  } else
    data_ = other.data_;
}

struct elem_size_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

size_t Scalar::size() const
{
  auto elem_size = type_dispatch(code_, elem_size_fn{});
  if (tuple_) {
    auto num_elements = *static_cast<const int32_t*>(data_);
    return sizeof(int32_t) + num_elements * elem_size;
  } else
    return elem_size;
}

}  // namespace legate
