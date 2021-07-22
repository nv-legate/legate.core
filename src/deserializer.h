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

#pragma once

#include <memory>

#include "legion.h"

#include "core/legate_defines.h"
#include "core/scalar.h"
#include "core/span.h"
#include "core/typedefs.h"

namespace legate {

template <class T>
class FromFuture {
 public:
  FromFuture()                   = default;
  FromFuture(const FromFuture &) = default;
  FromFuture(FromFuture &&)      = default;

  FromFuture &operator=(const FromFuture &) = default;
  FromFuture &operator=(FromFuture &&) = default;

  FromFuture(const T &value) : value_(value) {}
  FromFuture(T &&value) : value_(std::forward<T>(value)) {}

  inline operator T() const { return value(); }

  const T &value() const { return value_; }
  T &value() { return value_; }

 private:
  T value_;
};

// A class for helping with deserialization of arguments from python
class LegateDeserializer {
 public:
  LegateDeserializer(const void *a, size_t l) : args(static_cast<const char *>(a)), length(l) {}

 public:
  const void *data() const { return args; }
  void skip(size_t bytes) { args = args + bytes; }

 public:
  inline void check_type(int type_val, size_t type_size)
  {
    assert(length >= sizeof(int));
    int expected_type = *((const int *)args);
    length -= sizeof(int);
    args += sizeof(int);
    // The expected_type code is hardcoded in legate/core/legion.py
    assert(expected_type == type_val);
    assert(length >= type_size);
  }

 public:
  inline int16_t unpack_8bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT8, sizeof(int8_t));
    int8_t result = *((const int8_t *)args);
    length -= sizeof(int8_t);
    args += sizeof(int8_t);
    return result;
  }
  inline int16_t unpack_16bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT16, sizeof(int16_t));
    int16_t result = *((const int16_t *)args);
    length -= sizeof(int16_t);
    args += sizeof(int16_t);
    return result;
  }
  inline int32_t unpack_32bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT32, sizeof(int32_t));
    int32_t result = *((const int32_t *)args);
    length -= sizeof(int32_t);
    args += sizeof(int32_t);
    return result;
  }
  inline int64_t unpack_64bit_int(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_INT64, sizeof(int64_t));
    int64_t result = *((const int64_t *)args);
    length -= sizeof(int64_t);
    args += sizeof(int64_t);
    return result;
  }
  inline uint16_t unpack_8bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT8, sizeof(uint8_t));
    uint8_t result = *((const uint8_t *)args);
    length -= sizeof(uint8_t);
    args += sizeof(uint8_t);
    return result;
  }
  inline uint16_t unpack_16bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT16, sizeof(uint16_t));
    uint16_t result = *((const uint16_t *)args);
    length -= sizeof(uint16_t);
    args += sizeof(uint16_t);
    return result;
  }
  inline uint32_t unpack_32bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT32, sizeof(uint32_t));
    uint32_t result = *((const uint32_t *)args);
    length -= sizeof(uint32_t);
    args += sizeof(uint32_t);
    return result;
  }
  inline uint64_t unpack_64bit_uint(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_UINT64, sizeof(uint64_t));
    uint64_t result = *((const uint64_t *)args);
    length -= sizeof(uint64_t);
    args += sizeof(uint64_t);
    return result;
  }
  inline float unpack_float(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT32, sizeof(float));
    float result = *((const float *)args);
    length += sizeof(float);
    args += sizeof(float);
    return result;
  }
  inline double unpack_double(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT64, sizeof(double));
    double result = *((const double *)args);
    length -= sizeof(double);
    args += sizeof(double);
    return result;
  }
  inline bool unpack_bool(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_BOOL, sizeof(bool));
    bool result = *((const bool *)args);
    length -= sizeof(bool);
    args += sizeof(bool);
    return result;
  }
  inline __half unpack_half(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_FLOAT16, sizeof(__half));
    __half result = *((const __half *)args);
    length -= sizeof(__half);
    args += sizeof(__half);
    return result;
  }
  inline char unpack_char(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_TOTAL + 1, sizeof(char));
    char result = *((const char *)args);
    length -= sizeof(char);
    args += sizeof(char);
    return result;
  }
  inline complex<float> unpack_64bit_complex(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_COMPLEX64, sizeof(complex<float>));
    complex<float> result = *((const complex<float> *)args);
    length -= sizeof(complex<float>);
    args += sizeof(complex<float>);
    return result;
  }
  inline complex<double> unpack_128bit_complex(void)
  {
    if (TYPE_SAFE_LEGATE) check_type(LEGION_TYPE_COMPLEX128, sizeof(complex<double>));
    complex<double> result = *((const complex<double> *)args);
    length -= sizeof(complex<double>);
    args += sizeof(complex<double>);
    return result;
  }
  inline void *unpack_buffer(size_t buffer_size)
  {
    void *result = (void *)args;
    length -= buffer_size;
    args += buffer_size;
    return result;
  }
  inline LegateTypeCode unpack_dtype(void) { return safe_cast_type_code(unpack_32bit_int()); }

 public:
  static LegateTypeCode safe_cast_type_code(int32_t code);

 protected:
  const char *args;
  size_t length;
};

class FutureWrapper;
class RegionField;
class OutputRegionField;
class Store;
class StoreTransform;

class Deserializer {
 public:
  Deserializer(const Legion::Task *task, const std::vector<Legion::PhysicalRegion> &regions);

 public:
  friend void deserialize(Deserializer &ctx, __half &value);
  friend void deserialize(Deserializer &ctx, float &value);
  friend void deserialize(Deserializer &ctx, double &value);
  friend void deserialize(Deserializer &ctx, std::uint64_t &value);
  friend void deserialize(Deserializer &ctx, std::uint32_t &value);
  friend void deserialize(Deserializer &ctx, std::uint16_t &value);
  friend void deserialize(Deserializer &ctx, std::uint8_t &value);
  friend void deserialize(Deserializer &ctx, std::int64_t &value);
  friend void deserialize(Deserializer &ctx, std::int32_t &value);
  friend void deserialize(Deserializer &ctx, std::int16_t &value);
  friend void deserialize(Deserializer &ctx, std::int8_t &value);
  friend void deserialize(Deserializer &ctx, bool &value);

 public:
  friend void deserialize(Deserializer &ctx, LegateTypeCode &code);

 public:
  friend void deserialize(Deserializer &ctx, Legion::DomainPoint &value);
  friend void deserialize(Deserializer &ctx, Scalar &value);
  friend void deserialize(Deserializer &ctx, FutureWrapper &value);
  friend void deserialize(Deserializer &ctx, RegionField &value);
  friend void deserialize(Deserializer &ctx, OutputRegionField &value);
  friend void deserialize(Deserializer &ctx, Store &store);
  friend std::unique_ptr<StoreTransform> deserialize_transform(Deserializer &ctx);

 public:
  template <class T>
  friend void deserialize(Deserializer &ctx,
                          std::vector<T> &vec,
                          bool resize     = true,
                          bool boxed_size = true)
  {
    if (resize) {
      if (boxed_size) {
        Scalar size;
        deserialize(ctx, size);
        vec.resize(size.value<uint32_t>());
      } else {
        auto size = ctx.deserializer_.unpack_32bit_uint();
        vec.resize(size);
      }
    }
    for (auto &v : vec) deserialize(ctx, v);
  }
  template <class T>
  friend void deserialize(Deserializer &ctx, FromFuture<T> &scalar)
  {
    // grab the scalar out of the first future
    scalar = FromFuture<T>{ctx.futures_[0].get_result<T>()};

    // discard the first future
    ctx.futures_ = ctx.futures_.subspan(1);
  }

 private:
  const Legion::Task *task_;
  Span<const Legion::PhysicalRegion> regions_;
  Span<const Legion::Future> futures_;
  LegateDeserializer deserializer_;
  std::vector<Legion::OutputRegion> outputs_;
};

}  // namespace legate
