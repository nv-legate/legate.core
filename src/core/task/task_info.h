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

#include "core/task/variant_options.h"
#include "core/utilities/typedefs.h"

namespace legate {

struct VariantInfo {
  VariantImpl body;
  Legion::CodeDescriptor code_desc;
  VariantOptions options;
};

class TaskInfo {
 public:
  TaskInfo(const std::string& task_name);

 public:
  const std::string& name() const { return task_name_; }

 public:
  void add_variant(LegateVariantCode vid,
                   VariantImpl body,
                   const Legion::CodeDescriptor& code_desc,
                   const VariantOptions& options);
  const VariantInfo* find_variant(LegateVariantCode vid) const;
  bool has_variant(LegateVariantCode vid) const;

 public:
  void register_task(Legion::TaskID task_id);

 private:
  friend std::ostream& operator<<(std::ostream& os, const TaskInfo& info);
  std::string task_name_;
  std::map<LegateVariantCode, VariantInfo> variants_;
};

}  // namespace legate
