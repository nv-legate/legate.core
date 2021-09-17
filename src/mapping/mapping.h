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

#include "mapping/task.h"

namespace legate {
namespace mapping {

enum class TaskTarget : int32_t {
  CPU = 1,
  GPU = 2,
  OMP = 3,
};

enum class StoreTarget : int32_t {
  SYSMEM    = 1,
  FBMEM     = 2,
  ZCMEM     = 3,
  SOCKETMEM = 4,
};

enum class AllocPolicy : int32_t {
  CAN_ALLOC   = 1,
  MUST_ALLOC  = 2,
  NEVER_ALLOC = 3,
};

enum class InstLayout : int32_t {
  SOA = 1,
  AOS = 2,
};

struct StoreMapping {
 public:
  std::vector<int32_t> ordering{};
  std::vector<Store> colocate{};
  StoreTarget target{StoreTarget::SYSMEM};
  AllocPolicy policy{AllocPolicy::CAN_ALLOC};
  InstLayout layout{InstLayout::SOA};
  bool exact{false};

 public:
  StoreMapping() {}

 public:
  StoreMapping(const StoreMapping&) = default;
  StoreMapping& operator=(const StoreMapping&) = default;

 public:
  StoreMapping(StoreMapping&&) = default;
  StoreMapping& operator=(StoreMapping&&) = default;

 public:
  static StoreMapping default_mapping(const Store& store, StoreTarget target, bool exact = false);
};

struct LegateMapper {
  virtual bool is_pure() const                                                              = 0;
  virtual TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options)  = 0;
  virtual std::vector<StoreMapping> store_mappings(const Task& task,
                                                   const std::vector<StoreTarget>& options) = 0;
  virtual Scalar tunable_value(TunableID tunable_id)                                        = 0;
};

}  // namespace mapping
}  // namespace legate
