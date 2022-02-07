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
BaseDeserializer<Deserializer>::BaseDeserializer(const Legion::Task* task)
  : task_(task), task_args_{static_cast<const int8_t*>(task->args), task->arglen}
{
}

template <typename Deserializer>
void BaseDeserializer<Deserializer>::_unpack(LegateTypeCode& value)
{
  value = static_cast<LegateTypeCode>(unpack<int32_t>());
}

template <typename Deserializer>
void BaseDeserializer<Deserializer>::_unpack(Scalar& value)
{
  auto tuple = unpack<bool>();
  auto code  = unpack<LegateTypeCode>();
  value      = Scalar(tuple, code, task_args_.ptr());
  task_args_ = task_args_.subspan(value.size());
}

template <typename Deserializer>
std::shared_ptr<StoreTransform> BaseDeserializer<Deserializer>::unpack_transform()
{
  auto code = unpack<int32_t>();
  switch (code) {
    case -1: {
      return nullptr;
    }
    case LEGATE_CORE_TRANSFORM_SHIFT: {
      auto dim    = unpack<int32_t>();
      auto offset = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_shared<Shift>(dim, offset, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROMOTE: {
      auto extra_dim = unpack<int32_t>();
      auto dim_size  = unpack<int64_t>();
      auto parent    = unpack_transform();
      return std::make_shared<Promote>(extra_dim, dim_size, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_PROJECT: {
      auto dim    = unpack<int32_t>();
      auto coord  = unpack<int64_t>();
      auto parent = unpack_transform();
      return std::make_shared<Project>(dim, coord, std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_TRANSPOSE: {
      auto axes   = unpack<std::vector<int32_t>>();
      auto parent = unpack_transform();
      return std::make_shared<Transpose>(std::move(axes), std::move(parent));
    }
    case LEGATE_CORE_TRANSFORM_DELINEARIZE: {
      auto dim    = unpack<int32_t>();
      auto sizes  = unpack<std::vector<int64_t>>();
      auto parent = unpack_transform();
      return std::make_shared<Delinearize>(dim, std::move(sizes), std::move(parent));
    }
  }
  assert(false);
  return nullptr;
}

}  // namespace legate
