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

#include <assert.h>

#include "core/mapping/grid.h"

namespace legate {
namespace mapping {

void Grid::initialize(std::vector<int32_t>&& gr)
{
  grid         = std::move(gr);
  int32_t ndim = static_cast<int32_t>(grid.size());
  pitches.resize(ndim, 1);
  for (int32_t dim = 1; dim < ndim; ++dim) pitches[dim] = pitches[dim - 1] * grid[dim - 1];
}

Grid Grid::tile(const Grid& subgrid) const
{
  assert(subgrid.grid.size() == grid.size());

  int32_t ndim = static_cast<int32_t>(grid.size());
  std::vector<int32_t> new_grid;
  for (int32_t dim = 0; dim < ndim; ++dim) {
    assert(grid[dim] >= subgrid.grid[dim] && grid[dim] % subgrid.grid[dim] == 0);
    new_grid.push_back(grid[dim] / subgrid.grid[dim]);
  }

  Grid result;
  result.initialize(std::move(new_grid));
  return std::move(result);
}

}  // namespace mapping
}  // namespace legate
