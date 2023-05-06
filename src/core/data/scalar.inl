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
  return *static_cast<const VAL*>(data_);
}

template <>
inline std::string Scalar::value() const
{
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
    auto size = static_cast<const FixedArrayType*>(type_.get())->num_elements();
    return Span<const VAL>(reinterpret_cast<const VAL*>(data_), size);
  } else
    return Span<const VAL>(static_cast<const VAL*>(data_), 1);
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
