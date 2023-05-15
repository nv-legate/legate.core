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

namespace legate {

/**
 * @ingroup runtime
 * @brief POD for library configuration.
 */
struct ResourceConfig {
  /**
   * @brief Maximum number of tasks that the library can register
   */
  int64_t max_tasks{1024};
  /**
   * @brief Maximum number of custom reduction operators that the library can register
   */
  int64_t max_reduction_ops{0};
  int64_t max_projections{0};
  int64_t max_shardings{0};
};

class ResourceIdScope {
 public:
  ResourceIdScope() = default;
  ResourceIdScope(int64_t base, int64_t size) : base_(base), size_(size) {}

 public:
  ResourceIdScope(const ResourceIdScope&) = default;

 public:
  int64_t translate(int64_t local_resource_id) const { return base_ + local_resource_id; }
  int64_t invert(int64_t resource_id) const
  {
    assert(in_scope(resource_id));
    return resource_id - base_;
  }

 public:
  bool valid() const { return base_ != -1; }
  bool in_scope(int64_t resource_id) const
  {
    return base_ <= resource_id && resource_id < base_ + size_;
  }
  int64_t size() const { return size_; }

 private:
  int64_t base_{-1};
  int64_t size_{-1};
};

}  // namespace legate
