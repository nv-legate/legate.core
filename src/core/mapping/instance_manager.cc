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

#include "core/mapping/instance_manager.h"
#include "core/utilities/dispatch.h"

namespace legate {
namespace mapping {

using namespace Legion;
using namespace Legion::Mapping;

using RegionGroupP = std::shared_ptr<RegionGroup>;

static Legion::Logger log_instmgr("instmgr");

RegionGroup::RegionGroup(const std::set<Region>& rs, const Domain bound)
  : regions(rs), bounding_box(bound)
{
}

RegionGroup::RegionGroup(std::set<Region>&& rs, const Domain bound)
  : regions(std::forward<decltype(regions)>(rs)), bounding_box(bound)
{
}

std::vector<LogicalRegion> RegionGroup::get_regions() const
{
  std::vector<LogicalRegion> result;
  result.insert(result.end(), regions.begin(), regions.end());
  return std::move(result);
}

std::ostream& operator<<(std::ostream& os, const RegionGroup& region_group)
{
  os << "RegionGroup(" << region_group.bounding_box << ": {";
  for (const auto& region : region_group.regions) os << region << ",";
  os << "})";
  return os;
}

bool InstanceSet::find_instance(Region region,
                                Instance& result,
                                const InstanceMappingPolicy& policy) const
{
  auto finder = groups_.find(region);
  if (finder == groups_.end()) return false;

  auto& group = finder->second;
  if (policy.exact && group->regions.size() > 1) return false;

  auto ifinder = instances_.find(group);
  assert(ifinder != instances_.end());

  auto& spec = ifinder->second;
  if (spec.policy == policy) {
    result = spec.instance;
    return true;
  } else
    return false;
}

// We define "too big" as the size of the "unused" points being bigger than the intersection
static inline bool too_big(size_t union_volume,
                           size_t my_volume,
                           size_t group_volume,
                           size_t intersect_volume)
{
  return (union_volume - (my_volume + group_volume - intersect_volume)) > intersect_volume;
}

struct construct_overlapping_region_group_fn {
  template <int32_t DIM>
  RegionGroupP operator()(
    const InstanceSet::Region& region,
    const InstanceSet::Domain& domain,
    const std::map<InstanceSet::RegionGroupP, InstanceSet::InstanceSpec>& instances)
  {
    auto bound       = domain.bounds<DIM, coord_t>();
    size_t bound_vol = bound.volume();
    std::set<InstanceSet::Region> regions{region};

#ifdef DEBUG_LEGATE
    log_instmgr.debug() << "construct_overlapping_region_group( " << region << "," << domain << ")";
#endif

    for (const auto& pair : instances) {
      auto& group = pair.first;

      Rect<DIM> group_bbox = group->bounding_box.bounds<DIM, coord_t>();
#ifdef DEBUG_LEGATE
      log_instmgr.debug() << "  check intersection with " << group_bbox;
#endif
      auto intersect = bound.intersection(group_bbox);
      if (intersect.empty()) {
#ifdef DEBUG_LEGATE
        log_instmgr.debug() << "    no intersection";
#endif
        continue;
      }

      // Only allow merging if the bloating isn't "too big"
      auto union_bbox  = bound.union_bbox(group_bbox);
      size_t bound_vol = bound.volume();
      size_t union_vol = union_bbox.volume();
      if (too_big(union_vol, bound_vol, group_bbox.volume(), intersect.volume())) {
#ifdef DEBUG_LEGATE
        log_instmgr.debug() << "    too big to merge (union:" << union_bbox
                            << ",bound:" << bound_vol << ",group:" << group_bbox.volume()
                            << ",intersect:" << intersect.volume() << ")";
#endif
        continue;
      }

      regions.insert(group->regions.begin(), group->regions.end());
#ifdef DEBUG_LEGATE
      log_instmgr.debug() << "    bounds updated: " << bound << " ~> " << union_bbox;
#endif

      bound     = union_bbox;
      bound_vol = union_vol;
    }

    return std::make_shared<RegionGroup>(std::move(regions), InstanceSet::Domain(bound));
  }
};

RegionGroupP InstanceSet::construct_overlapping_region_group(const Region& region,
                                                             const Domain& domain,
                                                             bool exact) const
{
  auto finder = groups_.find(region);
  if (finder == groups_.end())
    return dim_dispatch(
      domain.get_dim(), construct_overlapping_region_group_fn{}, region, domain, instances_);
  else {
    if (!exact || finder->second->regions.size() == 1) return finder->second;
    return std::make_shared<RegionGroup>(std::set<Region>{region}, domain);
  }
}

std::set<InstanceSet::Instance> InstanceSet::record_instance(RegionGroupP group,
                                                             Instance instance,
                                                             const InstanceMappingPolicy& policy)
{
  std::set<Instance> replaced;
  std::set<RegionGroupP> removed_groups;

  auto finder = instances_.find(group);
  if (finder != instances_.end()) {
    replaced.insert(finder->second.instance);
    finder->second = InstanceSpec(instance, policy);
  } else
    instances_[group] = InstanceSpec(instance, policy);

  for (auto& region : group->regions) {
    auto finder = groups_.find(region);
    if (finder == groups_.end())
      groups_[region] = group;
    else if (finder->second != group) {
      // NOTE: This assumes that when a region changes groups, all other regions originally in the
      // same group also move to a new group. This is guaranteed in the BaseMapper because the new
      // group is synthesized within the same atomic block as the record_instance call we are in.
      removed_groups.insert(finder->second);
      finder->second = group;
    }
  }

  for (RegionGroupP removed_group : removed_groups) {
    replaced.insert(instances_[removed_group].instance);
    instances_.erase(removed_group);
  }

  replaced.erase(instance);
  return std::move(replaced);
}

bool InstanceSet::erase(PhysicalInstance inst)
{
  std::set<RegionGroupP> filtered_groups;
  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      auto to_erase = it++;
      filtered_groups.insert(to_erase->first);
      instances_.erase(to_erase);
    } else
      it++;
  }

  for (auto& group : filtered_groups)
    for (auto& region : group->regions) groups_.erase(region);

  return instances_.empty();
}

size_t InstanceSet::get_instance_size() const
{
  size_t sum = 0;
  for (auto& pair : instances_) sum += pair.second.instance.get_instance_size();
  return sum;
}

bool InstanceManager::find_instance(Region region,
                                    FieldID field_id,
                                    Memory memory,
                                    Instance& result,
                                    const InstanceMappingPolicy& policy)
{
  auto finder = instance_sets_.find(FieldMemInfo(region.get_tree_id(), field_id, memory));
  return policy.allocation != AllocPolicy::MUST_ALLOC && finder != instance_sets_.end() &&
         finder->second.find_instance(region, result, policy);
}

RegionGroupP InstanceManager::find_region_group(const Region& region,
                                                const Domain& domain,
                                                FieldID field_id,
                                                Memory memory,
                                                bool exact /*=false*/)
{
  FieldMemInfo key(region.get_tree_id(), field_id, memory);

  RegionGroupP result{nullptr};

  auto finder = instance_sets_.find(key);
  if (finder == instance_sets_.end() || exact)
    result = std::make_shared<RegionGroup>(std::set<Region>{region}, domain);
  else
    result = finder->second.construct_overlapping_region_group(region, domain, exact);

#ifdef DEBUG_LEGATE
  log_instmgr.debug() << "find_region_group(" << region << "," << domain << "," << field_id << ","
                      << memory << "," << exact << ") ~> " << *result;
#endif

  return std::move(result);
}

std::set<InstanceManager::Instance> InstanceManager::record_instance(
  RegionGroupP group, FieldID fid, Instance instance, const InstanceMappingPolicy& policy)
{
  const auto mem = instance.get_location();
  const auto tid = instance.get_tree_id();

  FieldMemInfo key(tid, fid, mem);
  return instance_sets_[key].record_instance(group, instance, policy);
}

void InstanceManager::erase(PhysicalInstance inst)
{
  const auto mem = inst.get_location();
  const auto tid = inst.get_tree_id();

  for (auto fit = instance_sets_.begin(); fit != instance_sets_.end(); /*nothing*/) {
    if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
      fit++;
      continue;
    }
    if (fit->second.erase(inst)) {
      auto to_erase = fit++;
      instance_sets_.erase(to_erase);
    } else
      fit++;
  }
}

std::map<Legion::Memory, size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Legion::Memory, size_t> result;
  for (auto& pair : instance_sets_) {
    auto& memory = pair.first.memory;
    if (result.find(memory) == result.end()) result[memory] = 0;
    result[memory] += pair.second.get_instance_size();
  }
  return result;
}

void InstanceManager::lock() { manager_lock_.lock(); }

void InstanceManager::unlock() { manager_lock_.unlock(); }

/*static*/ InstanceManager* InstanceManager::get_instance_manager()
{
  static InstanceManager* manager{nullptr};

  if (nullptr == manager) manager = new InstanceManager();
  return manager;
}

}  // namespace mapping
}  // namespace legate
