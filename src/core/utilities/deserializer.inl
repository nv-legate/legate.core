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

template <typename Deserializer>
BaseDeserializer<Deserializer>::BaseDeserializer(const int8_t* args, size_t arglen)
  : args_(Span<const int8_t>(args, arglen))
{
}

template <typename Deserializer>
void BaseDeserializer<Deserializer>::_unpack(Scalar& value)
{
  auto tuple = unpack<bool>();
  auto type  = unpack_type();
  value      = Scalar(tuple, std::move(type), args_.ptr());
  args_      = args_.subspan(value.size());
}

template <typename Deserializer>
std::shared_ptr<TransformStack> BaseDeserializer<Deserializer>::unpack_transform()
{
  auto code = unpack<int32_t>();
  switch (code) {
    case -1: {
      return std::make_shared<TransformStack>();
    }
    case LEGATE_CORE_TRANSFORM_SHIFT: {
      auto dim    = unpack<int32_t>();
      auto offset = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_shared<TransformStack>(std::make_unique<Shift>(dim, offset),
                                              std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROMOTE: {
      auto extra_dim = unpack<int32_t>();
      auto dim_size  = unpack<int64_t>();
      auto parent    = unpack_transform();
      return std::make_shared<TransformStack>(std::make_unique<Promote>(extra_dim, dim_size),
                                              std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROJECT: {
      auto dim    = unpack<int32_t>();
      auto coord  = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_shared<TransformStack>(std::make_unique<Project>(dim, coord),
                                              std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
      auto axes   = unpack<std::vector<int32_t>>();
      auto parent = unpack_transform();
      return std::make_shared<TransformStack>(std::make_unique<Transpose>(std::move(axes)),
                                              std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
      auto dim    = unpack<int32_t>();
      auto sizes  = unpack<std::vector<int64_t>>();
      auto parent = unpack_transform();
      return std::make_shared<TransformStack>(std::make_unique<Delinearize>(dim, std::move(sizes)),
                                              std::move(parent));
    }
  }
  assert(false);
  return nullptr;
}

template <typename Deserializer>
std::unique_ptr<Type> BaseDeserializer<Deserializer>::unpack_type()
{
  auto code = static_cast<Type::Code>(unpack<int32_t>());
  switch (code) {
    case Type::Code::FIXED_ARRAY: {
      auto uid  = unpack<int32_t>();
      auto N    = unpack<uint32_t>();
      auto type = unpack_type();
      return std::make_unique<FixedArrayType>(uid, std::move(type), N);
    }
    case Type::Code::STRUCT: {
      auto uid        = unpack<int32_t>();
      auto num_fields = unpack<uint32_t>();
      std::vector<std::unique_ptr<Type>> field_types(num_fields);
      for (auto& field_type : field_types) field_type = unpack_type();
      return std::make_unique<StructType>(uid, std::move(field_types));
    }
    case Type::Code::BOOL:
    case Type::Code::INT8:
    case Type::Code::INT16:
    case Type::Code::INT32:
    case Type::Code::INT64:
    case Type::Code::UINT8:
    case Type::Code::UINT16:
    case Type::Code::UINT32:
    case Type::Code::UINT64:
    case Type::Code::FLOAT16:
    case Type::Code::FLOAT32:
    case Type::Code::FLOAT64:
    case Type::Code::COMPLEX64:
    case Type::Code::COMPLEX128: {
      return std::make_unique<PrimitiveType>(code);
    }
  }
  LEGATE_ABORT;
  return nullptr;
}

}  // namespace legate
