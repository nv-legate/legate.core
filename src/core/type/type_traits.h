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

#include "core/type/type_info.h"

#include <limits.h>

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
#define COMPLEX_HALF
#endif
#include "mathtypes/complex.h"
#endif

#ifdef LEGION_REDOP_HALF
#include "mathtypes/half.h"
#endif

/**
 * @file
 * @brief Definitions for type traits in Legate
 */

namespace legate {

// This maps a type to its type code
#if defined(__clang__) && !defined(__NVCC__)
#define PREFIX static
#else
#define PREFIX
#endif

/**
 * @ingroup util
 * @brief A template constexpr that converts types to type codes
 */
template <class>
PREFIX constexpr Type::Code legate_type_code_of = Type::Code::INVALID;
template <>
PREFIX constexpr Type::Code legate_type_code_of<__half> = Type::Code::FLOAT16;
template <>
PREFIX constexpr Type::Code legate_type_code_of<float> = Type::Code::FLOAT32;
template <>
PREFIX constexpr Type::Code legate_type_code_of<double> = Type::Code::FLOAT64;
template <>
PREFIX constexpr Type::Code legate_type_code_of<int8_t> = Type::Code::INT8;
template <>
PREFIX constexpr Type::Code legate_type_code_of<int16_t> = Type::Code::INT16;
template <>
PREFIX constexpr Type::Code legate_type_code_of<int32_t> = Type::Code::INT32;
template <>
PREFIX constexpr Type::Code legate_type_code_of<int64_t> = Type::Code::INT64;
template <>
PREFIX constexpr Type::Code legate_type_code_of<uint8_t> = Type::Code::UINT8;
template <>
PREFIX constexpr Type::Code legate_type_code_of<uint16_t> = Type::Code::UINT16;
template <>
PREFIX constexpr Type::Code legate_type_code_of<uint32_t> = Type::Code::UINT32;
template <>
PREFIX constexpr Type::Code legate_type_code_of<uint64_t> = Type::Code::UINT64;
template <>
PREFIX constexpr Type::Code legate_type_code_of<bool> = Type::Code::BOOL;
template <>
PREFIX constexpr Type::Code legate_type_code_of<complex<float>> = Type::Code::COMPLEX64;
template <>
PREFIX constexpr Type::Code legate_type_code_of<complex<double>> = Type::Code::COMPLEX128;

#undef PREFIX

template <Type::Code CODE>
struct LegateTypeOf {
  using type = void;
};
template <>
struct LegateTypeOf<Type::Code::BOOL> {
  using type = bool;
};
template <>
struct LegateTypeOf<Type::Code::INT8> {
  using type = int8_t;
};
template <>
struct LegateTypeOf<Type::Code::INT16> {
  using type = int16_t;
};
template <>
struct LegateTypeOf<Type::Code::INT32> {
  using type = int32_t;
};
template <>
struct LegateTypeOf<Type::Code::INT64> {
  using type = int64_t;
};
template <>
struct LegateTypeOf<Type::Code::UINT8> {
  using type = uint8_t;
};
template <>
struct LegateTypeOf<Type::Code::UINT16> {
  using type = uint16_t;
};
template <>
struct LegateTypeOf<Type::Code::UINT32> {
  using type = uint32_t;
};
template <>
struct LegateTypeOf<Type::Code::UINT64> {
  using type = uint64_t;
};
template <>
struct LegateTypeOf<Type::Code::FLOAT16> {
  using type = __half;
};
template <>
struct LegateTypeOf<Type::Code::FLOAT32> {
  using type = float;
};
template <>
struct LegateTypeOf<Type::Code::FLOAT64> {
  using type = double;
};
template <>
struct LegateTypeOf<Type::Code::COMPLEX64> {
  using type = complex<float>;
};
template <>
struct LegateTypeOf<Type::Code::COMPLEX128> {
  using type = complex<double>;
};

/**
 * @ingroup util
 * @brief A template that converts type codes to types
 */
template <Type::Code CODE>
using legate_type_of = typename LegateTypeOf<CODE>::type;

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of an integral type
 */
template <Type::Code CODE>
struct is_integral {
  static constexpr bool value = std::is_integral<legate_type_of<CODE>>::value;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a signed integral type
 */
template <Type::Code CODE>
struct is_signed {
  static constexpr bool value = std::is_signed<legate_type_of<CODE>>::value;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of an unsigned integral type
 */
template <Type::Code CODE>
struct is_unsigned {
  static constexpr bool value = std::is_unsigned<legate_type_of<CODE>>::value;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a floating point type
 */
template <Type::Code CODE>
struct is_floating_point {
  static constexpr bool value = std::is_floating_point<legate_type_of<CODE>>::value;
};

template <>
struct is_floating_point<Type::Code::FLOAT16> {
  static constexpr bool value = true;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a complex type
 */
template <Type::Code CODE>
struct is_complex : std::false_type {};

template <>
struct is_complex<Type::Code::COMPLEX64> : std::true_type {};

template <>
struct is_complex<Type::Code::COMPLEX128> : std::true_type {};

/**
 * @ingroup util
 * @brief A predicate that holds if the type is one of the supported complex types
 */
template <typename T>
struct is_complex_type : std::false_type {};

template <>
struct is_complex_type<complex<float>> : std::true_type {};

template <>
struct is_complex_type<complex<double>> : std::true_type {};

}  // namespace legate
