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

#include <numeric>

#include "legate.h"

#include "core/mapping/mapping.h"

namespace legate {
namespace mapping {

DimOrdering::DimOrdering(Kind _kind, std::vector<int32_t>&& _dims)
  : kind(_kind), dims(std::forward<decltype(dims)>(_dims))
{
}

/*static*/ DimOrdering DimOrdering::c_order() { return DimOrdering(Kind::C); }

/*static*/ DimOrdering DimOrdering::fortran_order() { return DimOrdering(Kind::FORTRAN); }

/*static*/ DimOrdering DimOrdering::custom_order(std::vector<int32_t>&& dims)
{
  return DimOrdering(Kind::CUSTOM, std::forward<decltype(dims)>(dims));
}

Memory::Kind get_memory_kind(StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return Memory::Kind::SYSTEM_MEM;
    case StoreTarget::FBMEM: return Memory::Kind::GPU_FB_MEM;
    case StoreTarget::ZCMEM: return Memory::Kind::Z_COPY_MEM;
    case StoreTarget::SOCKETMEM: return Memory::Kind::SOCKET_MEM;
    default: LEGATE_ABORT;
  }
  assert(false);
  return Memory::Kind::SYSTEM_MEM;
}

bool DimOrdering::operator==(const DimOrdering& other) const
{
  return kind == other.kind && relative == other.relative && dims == other.dims;
}

void DimOrdering::populate_dimension_ordering(const Store& store,
                                              std::vector<Legion::DimensionKind>& ordering) const
{
  // TODO: We need to implement the relative dimension ordering
  assert(!relative);
  switch (kind) {
    case Kind::C: {
      auto dim = store.region_field().dim();
      for (int32_t idx = dim - 1; idx >= 0; --idx)
        ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
    case Kind::FORTRAN: {
      auto dim = store.region_field().dim();
      for (int32_t idx = 0; idx < dim; ++idx)
        ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
    case Kind::CUSTOM: {
      for (auto idx : dims) ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
  }
}

void DimOrdering::set_c_order() { kind = Kind::C; }

void DimOrdering::set_fortran_order() { kind = Kind::FORTRAN; }

void DimOrdering::set_custom_order(std::vector<int32_t>&& dims)
{
  kind = Kind::CUSTOM;
  dims = std::forward<std::vector<int32_t>&&>(dims);
}

bool InstanceMappingPolicy::operator==(const InstanceMappingPolicy& other) const
{
  return target == other.target && allocation == other.allocation && layout == other.layout &&
         exact == other.exact && ordering == other.ordering;
}

bool InstanceMappingPolicy::operator!=(const InstanceMappingPolicy& other) const
{
  return !operator==(other);
}

bool InstanceMappingPolicy::subsumes(const InstanceMappingPolicy& other) const
{
  // the allocation policy doesn't concern the instance layout
  return target == other.target && layout == other.layout && ordering == other.ordering &&
         (exact || !other.exact);
}

void InstanceMappingPolicy::populate_layout_constraints(
  const Store& store, Legion::LayoutConstraintSet& layout_constraints) const
{
  std::vector<Legion::DimensionKind> dimension_ordering{};

  if (layout == InstLayout::AOS) dimension_ordering.push_back(DIM_F);
  ordering.populate_dimension_ordering(store, dimension_ordering);
  if (layout == InstLayout::SOA) dimension_ordering.push_back(DIM_F);

  layout_constraints.add_constraint(
    Legion::OrderingConstraint(dimension_ordering, false /*contiguous*/));

  layout_constraints.add_constraint(Legion::MemoryConstraint(get_memory_kind(target)));
}

/*static*/ InstanceMappingPolicy InstanceMappingPolicy::default_policy(StoreTarget target,
                                                                       bool exact)
{
  InstanceMappingPolicy policy{};
  policy.target = target;
  policy.exact  = exact;
  return std::move(policy);
}

bool StoreMapping::for_future() const
{
  for (auto& store : stores) return store.get().is_future();
  assert(false);
  return false;
}

bool StoreMapping::for_unbound_store() const
{
  for (auto& store : stores) return store.get().unbound();
  assert(false);
  return false;
}

const Store& StoreMapping::store() const
{
#ifdef DEBUG_LEGATE
  assert(stores.size() == 1);
#endif
  return stores.front();
}

uint32_t StoreMapping::requirement_index() const
{
#ifdef DEBUG_LEGATE
  assert(stores.size() > 0);
  uint32_t result = -1U;
  for (auto& store : stores) {
    auto idx = store.get().requirement_index();
    assert(result == -1U || result == idx);
    result = idx;
  }
  return result;
#else
  static constexpr uint32_t invalid = -1U;
  if (stores.empty()) return invalid;
  return stores.front().get().requirement_index();
#endif
}

std::set<uint32_t> StoreMapping::requirement_indices() const
{
  std::set<uint32_t> indices;
  for (auto& store : stores) {
    if (store.get().is_future()) continue;
    indices.insert(store.get().region_field().index());
  }
  return std::move(indices);
}

std::set<const Legion::RegionRequirement*> StoreMapping::requirements() const
{
  std::set<const Legion::RegionRequirement*> reqs;
  for (auto& store : stores) {
    if (store.get().is_future()) continue;
    auto* req = store.get().region_field().get_requirement();
    if (!req->region.exists()) continue;
    reqs.insert(req);
  }
  return std::move(reqs);
}

void StoreMapping::populate_layout_constraints(
  Legion::LayoutConstraintSet& layout_constraints) const
{
  policy.populate_layout_constraints(stores.front(), layout_constraints);

  std::vector<Legion::FieldID> fields{};
  if (stores.size() > 1) {
    std::set<Legion::FieldID> field_set{};
    for (auto& store : stores) {
      auto field_id = store.get().region_field().field_id();
      if (field_set.find(field_id) == field_set.end()) {
        fields.push_back(field_id);
        field_set.insert(field_id);
      }
    }
  } else
    fields.push_back(stores.front().get().region_field().field_id());
  layout_constraints.add_constraint(
    Legion::FieldConstraint(fields, false /*contiguous*/, false /*inorder*/));
}

/*static*/ StoreMapping StoreMapping::default_mapping(const Store& store,
                                                      StoreTarget target,
                                                      bool exact)
{
  StoreMapping mapping{};
  mapping.policy = InstanceMappingPolicy::default_policy(target, exact);
  mapping.stores.emplace_back(store);
  return std::move(mapping);
}

}  // namespace mapping
}  // namespace legate
