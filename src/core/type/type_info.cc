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

#include <numeric>
#include <unordered_map>

#include "core/runtime/runtime.h"
#include "core/type/type_info.h"
#include "core/type/type_traits.h"

namespace legate {

namespace {

const std::unordered_map<Type::Code, uint32_t> SIZEOF = {
  {Type::Code::BOOL, sizeof(legate_type_of<Type::Code::BOOL>)},
  {Type::Code::INT8, sizeof(legate_type_of<Type::Code::INT8>)},
  {Type::Code::INT16, sizeof(legate_type_of<Type::Code::INT16>)},
  {Type::Code::INT32, sizeof(legate_type_of<Type::Code::INT32>)},
  {Type::Code::INT64, sizeof(legate_type_of<Type::Code::INT64>)},
  {Type::Code::UINT8, sizeof(legate_type_of<Type::Code::UINT8>)},
  {Type::Code::UINT16, sizeof(legate_type_of<Type::Code::UINT16>)},
  {Type::Code::UINT32, sizeof(legate_type_of<Type::Code::UINT32>)},
  {Type::Code::UINT64, sizeof(legate_type_of<Type::Code::UINT64>)},
  {Type::Code::FLOAT16, sizeof(legate_type_of<Type::Code::FLOAT16>)},
  {Type::Code::FLOAT32, sizeof(legate_type_of<Type::Code::FLOAT32>)},
  {Type::Code::FLOAT64, sizeof(legate_type_of<Type::Code::FLOAT64>)},
  {Type::Code::COMPLEX64, sizeof(legate_type_of<Type::Code::COMPLEX64>)},
  {Type::Code::COMPLEX128, sizeof(legate_type_of<Type::Code::COMPLEX128>)},
};

const std::unordered_map<Type::Code, std::string> TYPE_NAMES = {
  {Type::Code::BOOL, "bool"},
  {Type::Code::INT8, "int8"},
  {Type::Code::INT16, "int16"},
  {Type::Code::INT32, "int32"},
  {Type::Code::INT64, "int64"},
  {Type::Code::UINT8, "uint8"},
  {Type::Code::UINT16, "uint16"},
  {Type::Code::UINT32, "uint32"},
  {Type::Code::UINT64, "uint64"},
  {Type::Code::FLOAT16, "float16"},
  {Type::Code::FLOAT32, "float32"},
  {Type::Code::FLOAT64, "float64"},
  {Type::Code::COMPLEX64, "complex64"},
  {Type::Code::COMPLEX128, "complex128"},
};

const char* _VARIABLE_SIZE_ERROR_MESSAGE = "Variable-size element type cannot be used";

}  // namespace

Type::Type(Code c) : code(c) {}

void Type::record_reduction_operator(int32_t op_kind, int32_t global_op_id) const
{
  Runtime::get_runtime()->record_reduction_operator(uid(), op_kind, global_op_id);
}

int32_t Type::find_reduction_operator(int32_t op_kind) const
{
  return Runtime::get_runtime()->find_reduction_operator(uid(), op_kind);
}

PrimitiveType::PrimitiveType(Code code) : Type(code), size_(SIZEOF.at(code)) {}

int32_t PrimitiveType::uid() const { return static_cast<int32_t>(code); }

std::unique_ptr<Type> PrimitiveType::clone() const { return std::make_unique<PrimitiveType>(code); }

std::string PrimitiveType::to_string() const { return TYPE_NAMES.at(code); }

ExtensionType::ExtensionType(int32_t uid, Type::Code code) : Type(code), uid_(uid) {}

FixedArrayType::FixedArrayType(int32_t uid,
                               std::unique_ptr<Type> element_type,
                               uint32_t N) noexcept(false)
  : ExtensionType(uid, Type::Code::FIXED_ARRAY),
    element_type_(std::move(element_type)),
    N_(N),
    size_(element_type_->size() * N)
{
  if (element_type_->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
}

std::unique_ptr<Type> FixedArrayType::clone() const
{
  return std::make_unique<FixedArrayType>(uid_, element_type_->clone(), N_);
}

std::string FixedArrayType::to_string() const
{
  std::stringstream ss;
  ss << element_type_->to_string() << "[" << N_ << "]";
  return std::move(ss).str();
}

StructType::StructType(int32_t uid,
                       std::vector<std::unique_ptr<Type>>&& field_types,
                       bool align) noexcept(false)
  : ExtensionType(uid, Type::Code::STRUCT),
    aligned_(align),
    alignment_(1),
    size_(0),
    field_types_(std::forward<decltype(field_types_)>(field_types))
{
  offsets_.reserve(field_types_.size());
  if (aligned_) {
    static constexpr auto align_offset = [](uint32_t offset, uint32_t align) {
      return (offset + (align - 1)) & -align;
    };

    for (auto& field_type : field_types_) {
      if (field_type->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
      uint32_t _my_align = field_type->alignment();
      alignment_         = std::max(_my_align, alignment_);

      uint32_t offset = align_offset(size_, _my_align);
      offsets_.push_back(offset);
      size_ = offset + field_type->size();
    }
    size_ = align_offset(size_, alignment_);
  } else {
    for (auto& field_type : field_types_) {
      if (field_type->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
      offsets_.push_back(size_);
      size_ += field_type->size();
    }
  }
}

std::unique_ptr<Type> StructType::clone() const
{
  std::vector<std::unique_ptr<Type>> field_types;
  for (auto& field_type : field_types_) field_types.push_back(field_type->clone());
  return std::make_unique<StructType>(uid_, std::move(field_types), aligned_);
}

std::string StructType::to_string() const
{
  std::stringstream ss;
  ss << "{";
  for (uint32_t idx = 0; idx < field_types_.size(); ++idx) {
    if (idx > 0) ss << ",";
    ss << field_types_.at(idx)->to_string() << ":" << offsets_.at(idx);
  }
  ss << "}";
  return std::move(ss).str();
}

const Type& StructType::field_type(uint32_t field_idx) const { return *field_types_.at(field_idx); }

StringType::StringType() : Type(Type::Code::STRING) {}

int32_t StringType::uid() const { return static_cast<int32_t>(code); }

std::unique_ptr<Type> StringType::clone() const { return string_type(); }

std::string StringType::to_string() const { return "string"; }

std::unique_ptr<Type> primitive_type(Type::Code code)
{
  return std::make_unique<PrimitiveType>(code);
}

std::unique_ptr<Type> string_type() { return std::make_unique<StringType>(); }

std::unique_ptr<Type> fixed_array_type(std::unique_ptr<Type> element_type,
                                       uint32_t N) noexcept(false)
{
  return std::make_unique<FixedArrayType>(
    Runtime::get_runtime()->get_type_uid(), std::move(element_type), N);
}

std::unique_ptr<Type> struct_type(std::vector<std::unique_ptr<Type>>&& field_types,
                                  bool align) noexcept(false)
{
  return std::make_unique<StructType>(Runtime::get_runtime()->get_type_uid(),
                                      std::forward<std::vector<std::unique_ptr<Type>>>(field_types),
                                      align);
}

std::ostream& operator<<(std::ostream& ostream, const Type::Code& code)
{
  ostream << static_cast<int32_t>(code);
  return ostream;
}

std::ostream& operator<<(std::ostream& ostream, const Type& type)
{
  ostream << type.to_string();
  return ostream;
}

}  // namespace legate
