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

#include <numeric>

#include "mapping/mapping.h"

using namespace Legion;

namespace legate {
namespace mapping {

bool StoreMapping::for_unbound_stores() const
{
  for (auto& store : colocate) return store.unbound();
  assert(false);
  return false;
}

uint32_t StoreMapping::requirement_index() const
{
  assert(colocate.size() > 0);
  uint32_t result = -1U;
  for (auto& store : colocate) {
    auto idx = store.region_field().index();
    assert(result == -1U || result == idx);
    result = idx;
  }
  return result;
}

void StoreMapping::populate_layout_constraints(Legion::LayoutConstraintSet& layout_constraints,
                                               const Legion::RegionRequirement& requirement) const
{
  if (requirement.redop > 0)
    layout_constraints.add_constraint(
      SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, requirement.redop));

  std::vector<DimensionKind> dimension_ordering{};
  if (layout == InstLayout::AOS) dimension_ordering.push_back(DIM_F);
  for (auto it = ordering.rbegin(); it != ordering.rend(); ++it)
    dimension_ordering.push_back(static_cast<DimensionKind>(DIM_X + *it));
  if (layout == InstLayout::SOA) dimension_ordering.push_back(DIM_F);
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, false /*contiguous*/));

  std::vector<FieldID> fields{};
  for (auto& store : colocate) fields.push_back(store.region_field().field_id());
  layout_constraints.add_constraint(FieldConstraint(fields, true /*contiguous*/));
}

/*static*/ StoreMapping StoreMapping::default_mapping(const Store& store,
                                                      StoreTarget target,
                                                      bool exact)
{
  StoreMapping mapping{};

  if (!store.unbound()) {
    mapping.ordering.resize(store.dim());
    std::iota(mapping.ordering.begin(), mapping.ordering.end(), 0);
  }

  mapping.colocate.push_back(store);
  mapping.target = target;
  mapping.exact  = exact;

  return std::move(mapping);
}

}  // namespace mapping
}  // namespace legate
