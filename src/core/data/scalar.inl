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

namespace legate {

template <typename T>
Scalar::Scalar(T value) : own_(true), type_(primitive_type(legate_type_code_of<T>))
{
  static_assert(legate_type_code_of<T> != Type::Code::FIXED_ARRAY);
  static_assert(legate_type_code_of<T> != Type::Code::STRUCT);
  static_assert(legate_type_code_of<T> != Type::Code::STRING);
  static_assert(legate_type_code_of<T> != Type::Code::INVALID);
  auto buffer = malloc(sizeof(T));
  memcpy(buffer, &value, sizeof(T));
  data_ = buffer;
}

template <typename T>
Scalar::Scalar(T value, std::unique_ptr<Type> type) : own_(true), type_(std::move(type))
{
  if (type_->code == Type::Code::INVALID)
    throw std::invalid_argument("Invalid type cannot be used");
  if (type_->size() != sizeof(T))
    throw std::invalid_argument("Size of the value doesn't match with the type");
  auto buffer = malloc(sizeof(T));
  memcpy(buffer, &value, sizeof(T));
  data_ = buffer;
}

template <typename T>
Scalar::Scalar(const std::vector<T>& values)
  : own_(true), type_(fixed_array_type(primitive_type(legate_type_code_of<T>), values.size()))
{
  auto size   = type_->size();
  auto buffer = malloc(size);
  memcpy(buffer, values.data(), size);
  data_ = buffer;
}

template <typename VAL>
VAL Scalar::value() const
{
  if (sizeof(VAL) != type_->size())
    throw std::invalid_argument("Size of the scalar is " + std::to_string(type_->size()) +
                                ", but the requested type has size " + std::to_string(sizeof(VAL)));
  return *static_cast<const VAL*>(data_);
}

template <>
inline std::string Scalar::value() const
{
  if (type_->code != Type::Code::STRING)
    throw std::invalid_argument("Type of the scalar is not string");
  // Getting a span of a temporary scalar is illegal in general,
  // but we know this is safe as the span's pointer is held by this object.
  auto len          = *static_cast<const uint32_t*>(data_);
  const auto* begin = static_cast<const char*>(data_) + sizeof(uint32_t);
  const auto* end   = begin + len;
  return std::string(begin, end);
}

template <typename VAL>
Span<const VAL> Scalar::values() const
{
  if (type_->code == Type::Code::FIXED_ARRAY) {
    auto arr_type         = static_cast<const FixedArrayType*>(type_.get());
    const auto& elem_type = arr_type->element_type();
    if (sizeof(VAL) != elem_type.size())
      throw std::invalid_argument(
        "The scalar's element type has size " + std::to_string(elem_type.size()) +
        ", but the requested element type has size " + std::to_string(sizeof(VAL)));
    auto size = arr_type->num_elements();
    return Span<const VAL>(reinterpret_cast<const VAL*>(data_), size);
  } else {
    if (sizeof(VAL) != type_->size())
      throw std::invalid_argument("Size of the scalar is " + std::to_string(type_->size()) +
                                  ", but the requested element type has size " +
                                  std::to_string(sizeof(VAL)));
    return Span<const VAL>(static_cast<const VAL*>(data_), 1);
  }
}

template <>
inline Legion::DomainPoint Scalar::value<Legion::DomainPoint>() const
{
  Legion::DomainPoint result;
  auto span  = values<int64_t>();
  result.dim = span.size();
  for (auto idx = 0; idx < result.dim; ++idx) result[idx] = span[idx];
  return result;
}

}  // namespace legate
