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

#include "core/task/task_info.h"

namespace legate {

TaskInfo::TaskInfo(const std::string& task_name) : task_name_(task_name) {}

void TaskInfo::add_variant(LegateVariantCode vid, VariantImpl body, const VariantOptions& options)
{
#ifdef DEBUG_LEGATE
  assert(variants_.find(vid) == variants_.end());
#endif
  variants_.emplace(std::make_pair(vid, VariantInfo{body, options}));
}

const VariantInfo* TaskInfo::find_variant(LegateVariantCode vid) const
{
  auto finder = variants_.find(vid);
  return finder != variants_.end() ? &finder->second : nullptr;
}

bool TaskInfo::has_variant(LegateVariantCode vid) const
{
  return variants_.find(vid) != variants_.end();
}

std::ostream& operator<<(std::ostream& os, const VariantInfo& info)
{
  std::stringstream ss;
  ss << std::showbase << std::hex << reinterpret_cast<uintptr_t>(info.body) << "," << info.options;
  os << std::move(ss).str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const TaskInfo& info)
{
  static const char* variant_names[] = {"(invalid)", "CPU", "GPU", "OMP"};
  std::stringstream ss;
  ss << info.name() << " {";
  for (auto [vid, vi] : info.variants_) ss << variant_names[vid] << ":[" << vi << "],";
  ss << "}";
  os << std::move(ss).str();
  return os;
}

}  // namespace legate
