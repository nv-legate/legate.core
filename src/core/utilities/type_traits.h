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

#include "core/utilities/typedefs.h"

namespace legate {

// This maps a type to its LegateTypeCode
#if defined(__clang__) && !defined(__NVCC__)
template <class>
static constexpr LegateTypeCode legate_type_code_of = MAX_TYPE_NUMBER;

template <>
static constexpr LegateTypeCode legate_type_code_of<__half> = HALF_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<float> = FLOAT_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<double> = DOUBLE_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int8_t> = INT8_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int16_t> = INT16_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int32_t> = INT32_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<int64_t> = INT64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint8_t> = UINT8_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint16_t> = UINT16_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint32_t> = UINT32_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<uint64_t> = UINT64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<bool> = BOOL_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<complex<float>> = COMPLEX64_LT;
template <>
static constexpr LegateTypeCode legate_type_code_of<complex<double>> = COMPLEX128_LT;
#else  // not clang
template <class>
constexpr LegateTypeCode legate_type_code_of = MAX_TYPE_NUMBER;

template <>
constexpr LegateTypeCode legate_type_code_of<__half> = HALF_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<float> = FLOAT_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<double> = DOUBLE_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int8_t> = INT8_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int16_t> = INT16_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int32_t> = INT32_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<int64_t> = INT64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint8_t> = UINT8_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint16_t> = UINT16_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint32_t> = UINT32_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<uint64_t> = UINT64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<bool> = BOOL_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<complex<float>> = COMPLEX64_LT;
template <>
constexpr LegateTypeCode legate_type_code_of<complex<double>> = COMPLEX128_LT;
#endif

template <LegateTypeCode CODE>
struct LegateTypeOf {
  using type = void;
};
template <>
struct LegateTypeOf<LegateTypeCode::BOOL_LT> {
  using type = bool;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT8_LT> {
  using type = int8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT16_LT> {
  using type = int16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT32_LT> {
  using type = int32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::INT64_LT> {
  using type = int64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT8_LT> {
  using type = uint8_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT16_LT> {
  using type = uint16_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT32_LT> {
  using type = uint32_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::UINT64_LT> {
  using type = uint64_t;
};
template <>
struct LegateTypeOf<LegateTypeCode::HALF_LT> {
  using type = __half;
};
template <>
struct LegateTypeOf<LegateTypeCode::FLOAT_LT> {
  using type = float;
};
template <>
struct LegateTypeOf<LegateTypeCode::DOUBLE_LT> {
  using type = double;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX64_LT> {
  using type = complex<float>;
};
template <>
struct LegateTypeOf<LegateTypeCode::COMPLEX128_LT> {
  using type = complex<double>;
};

template <LegateTypeCode CODE>
using legate_type_of = typename LegateTypeOf<CODE>::type;

template <LegateTypeCode CODE>
struct is_integral {
  static constexpr bool value = std::is_integral<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_signed {
  static constexpr bool value = std::is_signed<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_unsigned {
  static constexpr bool value = std::is_unsigned<legate_type_of<CODE>>::value;
};

template <LegateTypeCode CODE>
struct is_floating_point {
  static constexpr bool value = std::is_floating_point<legate_type_of<CODE>>::value;
};

template <typename T>
struct is_complex : std::false_type {
};

template <>
struct is_complex<complex<float>> : std::true_type {
};

template <>
struct is_complex<complex<double>> : std::true_type {
};

}  // namespace legate
