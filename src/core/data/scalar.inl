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

namespace legate {

template <typename T>
Scalar::Scalar(T value) : tuple_(false), code_(legate_type_code_of<T>), data_(new T(value))
{
}

template <typename VAL>
VAL Scalar::value() const
{
  return *static_cast<const VAL*>(data_);
}

template <typename VAL>
Span<const VAL> Scalar::values() const
{
  if (tuple_) {
    auto size = *static_cast<const uint32_t*>(data_);
    auto data = static_cast<const uint8_t*>(data_) + sizeof(uint32_t);
    return Span<const VAL>(reinterpret_cast<const VAL*>(data), size);
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
