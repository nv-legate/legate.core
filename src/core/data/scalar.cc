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

Scalar::Scalar(const Scalar& other) : own_(other.own_), type_(other.type_->clone()) { copy(other); }

Scalar::Scalar(std::unique_ptr<Type> type, const void* data) : type_(std::move(type)), data_(data)
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
  own_  = other.own_;
  type_ = other.type_->clone();
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

size_t Scalar::size() const
{
  if (type_->code == Type::Code::STRING)
    return *static_cast<const uint32_t*>(data_) + sizeof(uint32_t);
  else
    return type_->size();
}

}  // namespace legate
