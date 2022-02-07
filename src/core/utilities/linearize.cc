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

#include "core/utilities/linearize.h"
#include "core/utilities/dispatch.h"

namespace legate {

using namespace Legion;

struct linearize_fn {
  template <int32_t DIM>
  size_t operator()(const DomainPoint& lo_dp, const DomainPoint& hi_dp, const DomainPoint& point_dp)
  {
    Point<DIM> lo      = lo_dp;
    Point<DIM> hi      = hi_dp;
    Point<DIM> point   = point_dp;
    Point<DIM> extents = hi - lo + Point<DIM>::ONES();
    size_t idx         = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) idx = idx * extents[dim] + point[dim] - lo[dim];
    return idx;
  }
};

size_t linearize(const DomainPoint& lo, const DomainPoint& hi, const DomainPoint& point)
{
  return dim_dispatch(point.dim, linearize_fn{}, lo, hi, point);
}

struct delinearize_fn {
  template <int32_t DIM>
  DomainPoint operator()(const DomainPoint& lo_dp, const DomainPoint& hi_dp, size_t idx)
  {
    Point<DIM> lo      = lo_dp;
    Point<DIM> hi      = hi_dp;
    Point<DIM> extents = hi - lo + Point<DIM>::ONES();
    Point<DIM> point;
    for (int32_t dim = DIM - 1; dim >= 0; --dim) {
      point[dim] = idx % extents[dim] + lo[dim];
      idx /= extents[dim];
    }
    return point;
  }
};

DomainPoint delinearize(const DomainPoint& lo, const DomainPoint& hi, size_t idx)
{
  return dim_dispatch(lo.dim, delinearize_fn{}, lo, hi, idx);
}

}  // namespace legate
