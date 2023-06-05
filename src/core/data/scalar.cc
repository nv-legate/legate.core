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

namespace legate {

Scalar::Scalar(const Scalar& other) : own_(other.own_), type_(other.type_->clone()) { copy(other); }

Scalar::Scalar(Scalar&& other) : own_(other.own_), type_(std::move(other.type_)), data_(other.data_)
{
  other.own_  = false;
  other.type_ = nullptr;
  other.data_ = nullptr;
}

Scalar::Scalar(std::unique_ptr<Type> type, const void* data) : type_(std::move(type)), data_(data)
{
}

Scalar::Scalar(const std::string& string) : own_(true), type_(string_type())
{
  auto data_size                  = sizeof(char) * string.size();
  auto buffer                     = malloc(sizeof(uint32_t) + data_size);
  *static_cast<uint32_t*>(buffer) = string.size();
  memcpy(static_cast<int8_t*>(buffer) + sizeof(uint32_t), string.data(), data_size);
  data_ = buffer;
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
