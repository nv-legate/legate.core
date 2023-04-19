/* Copyright 2023 NVIDIA Corporation
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

#include "core/legate_c.h"

#include <memory>
#include <vector>

namespace legate {

struct Type {
  enum class Code : int32_t {
    BOOL        = BOOL_LT,
    INT8        = INT8_LT,
    INT16       = INT16_LT,
    INT32       = INT32_LT,
    INT64       = INT64_LT,
    UINT8       = UINT8_LT,
    UINT16      = UINT16_LT,
    UINT32      = UINT32_LT,
    UINT64      = UINT64_LT,
    FLOAT16     = FLOAT16_LT,
    FLOAT32     = FLOAT32_LT,
    FLOAT64     = FLOAT64_LT,
    COMPLEX64   = COMPLEX64_LT,
    COMPLEX128  = COMPLEX128_LT,
    FIXED_ARRAY = FIXED_ARRAY_LT,
    STRUCT      = STRUCT_LT,
    STRING      = STRING_LT,
    INVALID     = INVALID_LT,
  };

  Type(Code code);
  virtual ~Type() {}

  virtual uint32_t size() const               = 0;
  virtual int32_t uid() const                 = 0;
  virtual bool variable_size() const          = 0;
  virtual std::unique_ptr<Type> clone() const = 0;
  virtual std::string to_string() const       = 0;

  const Code code;
};

class PrimitiveType : public Type {
 public:
  PrimitiveType(Code code);
  uint32_t size() const override { return size_; }
  int32_t uid() const override;
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;

 private:
  const uint32_t size_;
};

class ExtensionType : public Type {
 public:
  ExtensionType(int32_t uid, Type::Code code);
  int32_t uid() const override { return uid_; }

 protected:
  const uint32_t uid_;
};

class FixedArrayType : public ExtensionType {
 public:
  FixedArrayType(int32_t uid, std::unique_ptr<Type> element_type, uint32_t N);
  uint32_t size() const override { return size_; }
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  uint32_t num_elements() const { return N_; }
  const Type* element_type() const { return element_type_.get(); }

 private:
  const std::unique_ptr<Type> element_type_;
  const uint32_t N_;
  const uint32_t size_;
};

class StructType : public ExtensionType {
 public:
  StructType(int32_t uid, std::vector<std::unique_ptr<Type>>&& field_types);
  uint32_t size() const override;
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  uint32_t num_fields() const { return field_types_.size(); }
  const Type* field_type(uint32_t field_idx) const;

 private:
  std::vector<std::unique_ptr<Type>> field_types_{};
};

class StringType : public Type {
 public:
  StringType();
  bool variable_size() const override { return true; }
  uint32_t size() const override { return 0; }
  int32_t uid() const override;
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
};

std::unique_ptr<Type> primitive_type(Type::Code code);

std::unique_ptr<Type> string_type();

std::unique_ptr<Type> fixed_array_type(int32_t uid, std::unique_ptr<Type> element_type, uint32_t N);

std::unique_ptr<Type> struct_type(int32_t uid, std::vector<std::unique_ptr<Type>>&& field_types);

// The caller transfers ownership of the Type objects
std::unique_ptr<Type> struct_type_raw_ptrs(int32_t uid, std::vector<Type*> field_types);

}  // namespace legate
