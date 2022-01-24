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

#include <stdint.h>
#include <vector>

namespace legate {
namespace mapping {

class Grid {
 public:
  Grid() {}

 public:
  Grid(const Grid&) = default;
  Grid(Grid&&)      = default;

 public:
  Grid& operator=(const Grid&) = default;
  Grid& operator=(Grid&&) = default;

 public:
  void initialize(std::vector<int32_t>&& grid);
  Grid tile(const Grid& subgrid) const;

 public:
  std::vector<int32_t> grid{};
  std::vector<int32_t> pitches{};
};

}  // namespace mapping
}  // namespace legate
