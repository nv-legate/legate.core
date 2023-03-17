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

std::vector<RegionGroup::Region> RegionGroup::get_regions() const
{
  std::vector<Region> result;
  result.insert(result.end(), regions.begin(), regions.end());
  return std::move(result);
}

bool RegionGroup::subsumes(const RegionGroup* other)
{
  if (regions.size() < other->regions.size()) return false;
  if (other->regions.size() == 1) {
    return regions.find(*other->regions.begin()) != regions.end();
  } else {
    auto finder = subsumption_cache.find(other);
    if (finder != subsumption_cache.end()) return finder->second;
    for (auto& region : other->regions)
      if (regions.find(region) == regions.end()) {
        subsumption_cache[other] = false;
        return false;
      }

    subsumption_cache[other] = true;
    return true;
  }
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

  auto ifinder = instances_.find(group.get());
  assert(ifinder != instances_.end());

  auto& spec = ifinder->second;
  if (spec.policy.subsumes(policy)) {
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
  RegionGroupP operator()(const InstanceSet::Region& region,
                          const Domain& domain,
                          const std::map<RegionGroup*, InstanceSet::InstanceSpec>& instances)
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
      auto union_bbox      = bound.union_bbox(group_bbox);
      size_t union_vol     = union_bbox.volume();
      size_t group_vol     = group_bbox.volume();
      size_t intersect_vol = intersect.volume();
      if (too_big(union_vol, bound_vol, group_vol, intersect_vol)) {
#ifdef DEBUG_LEGATE
        log_instmgr.debug() << "    too big to merge (union:" << union_bbox
                            << ",bound:" << bound_vol << ",group:" << group_vol
                            << ",intersect:" << intersect_vol << ")";
#endif
        continue;
      }

      // NOTE: It is critical that we maintain the invariant that if at least one region is mapped
      // to a group in the instances_ table, that group is still present on the groups_ table, and
      // thus there's at least one shared_ptr remaining that points to it. Otherwise we run the risk
      // that a group pointer stored on the instances_ table points to a group that's been collected
      regions.insert(group->regions.begin(), group->regions.end());
#ifdef DEBUG_LEGATE
      log_instmgr.debug() << "    bounds updated: " << bound << " ~> " << union_bbox;
#endif

      bound     = union_bbox;
      bound_vol = union_vol;
    }

    return std::make_shared<RegionGroup>(std::move(regions), Domain(bound));
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
#ifdef DEBUG_LEGATE
#ifdef DEBUG_INSTANCE_MANAGER
  log_instmgr.debug() << "===== before adding an entry " << *group << " ~> " << instance
                      << " =====";
#endif
  dump_and_sanity_check();
#endif

  std::set<Instance> replaced;
  std::set<RegionGroupP> removed_groups;

  auto finder = instances_.find(group.get());
  if (finder != instances_.end()) {
    replaced.insert(finder->second.instance);
    finder->second = InstanceSpec(instance, policy);
  } else
    instances_[group.get()] = InstanceSpec(instance, policy);

  for (auto& region : group->regions) {
    auto finder = groups_.find(region);
    if (finder == groups_.end())
      groups_[region] = group;
    else if (finder->second != group) {
      removed_groups.insert(finder->second);
      finder->second = group;
    }
  }

  for (auto& removed_group : removed_groups) {
    // Because of exact policies, we can't simply remove the groups where regions in the `group`
    // originally belonged, because one region can be included in multiple region groups. (Note that
    // the exact mapping bypasses the coalescing heuristic and always creates a fresh singleton
    // group.) So, before we prune out each of those potentially obsolete groups, we need to
    // make sure that it has no remaining references.
    bool can_remove = true;
    for (Region rg : removed_group->regions) {
      if (groups_.at(rg) == removed_group) {
        can_remove = false;
        break;
      }
    }
    if (can_remove) {
      auto finder = instances_.find(removed_group.get());
      replaced.insert(finder->second.instance);
      instances_.erase(finder);
    }
  }

  replaced.erase(instance);

#ifdef DEBUG_LEGATE
#ifdef DEBUG_INSTANCE_MANAGER
  log_instmgr.debug() << "===== after adding an entry " << *group << " ~> " << instance << " =====";
#endif
  dump_and_sanity_check();
#endif

  return std::move(replaced);
}

bool InstanceSet::erase(Instance inst)
{
  std::set<RegionGroup*> filtered_groups;
#ifdef DEBUG_LEGATE
#ifdef DEBUG_INSTANCE_MANAGER
  log_instmgr.debug() << "===== before erasing an instance " << inst << " =====";
#endif
  dump_and_sanity_check();
#endif

  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      auto to_erase = it++;
      filtered_groups.insert(to_erase->first);
      instances_.erase(to_erase);
    } else
      it++;
  }

  std::set<Region> filtered_regions;
  for (RegionGroup* group : filtered_groups)
    for (Region region : group->regions)
      if (groups_.at(region).get() == group)
        // We have to do this in two steps; we don't want to remove the last shared_ptr to a group
        // while iterating over the same group's regions
        filtered_regions.insert(region);
  for (Region region : filtered_regions) groups_.erase(region);

#ifdef DEBUG_LEGATE
#ifdef DEBUG_INSTANCE_MANAGER
  log_instmgr.debug() << "===== after erasing an instance " << inst << " =====";
#endif
  dump_and_sanity_check();
#endif

  return instances_.empty();
}

size_t InstanceSet::get_instance_size() const
{
  size_t sum = 0;
  for (auto& pair : instances_) sum += pair.second.instance.get_instance_size();
  return sum;
}

void InstanceSet::dump_and_sanity_check() const
{
#ifdef DEBUG_INSTANCE_MANAGER
  for (auto& entry : groups_) log_instmgr.debug() << "  " << entry.first << " ~> " << *entry.second;
  for (auto& entry : instances_)
    log_instmgr.debug() << "  " << *entry.first << " ~> " << entry.second.instance;
#endif
  std::set<RegionGroup*> found_groups;
  for (auto& entry : groups_) {
    found_groups.insert(entry.second.get());
    assert(instances_.count(entry.second.get()) > 0);
    assert(entry.second->regions.count(entry.first) > 0);
  }
  for (auto& entry : instances_) assert(found_groups.count(entry.first) > 0);
}

bool ReductionInstanceSet::find_instance(ReductionOpID& redop,
                                         Region& region,
                                         Instance& result,
                                         const InstanceMappingPolicy& policy) const
{
  auto finder = instances_.find(region);
  if (finder == instances_.end()) return false;
  auto& spec = finder->second;
  if (spec.policy == policy && spec.redop == redop) {
    result = spec.instance;
    return true;
  } else
    return false;
}

void ReductionInstanceSet::record_instance(ReductionOpID& redop,
                                           Region& region,
                                           Instance& instance,
                                           const InstanceMappingPolicy& policy)
{
  auto finder = instances_.find(region);
  if (finder != instances_.end()) {
    auto& spec = finder->second;
    if (spec.policy != policy || spec.redop != redop) {
      instances_.insert_or_assign(region, ReductionInstanceSpec(redop, instance, policy));
    }
  } else {
    instances_[region] = ReductionInstanceSpec(redop, instance, policy);
  }
}

bool ReductionInstanceSet::erase(Instance inst)
{
  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      auto to_erase = it++;
      instances_.erase(to_erase);
    } else
      it++;
  }
  return instances_.empty();
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

void InstanceManager::erase(Instance inst)
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

std::map<Memory, size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Memory, size_t> result;
  for (auto& pair : instance_sets_) {
    auto& memory = pair.first.memory;
    if (result.find(memory) == result.end()) result[memory] = 0;
    result[memory] += pair.second.get_instance_size();
  }
  return result;
}

/*static*/ InstanceManager* InstanceManager::get_instance_manager()
{
  static InstanceManager* manager{nullptr};

  if (nullptr == manager) manager = new InstanceManager();
  return manager;
}

bool ReductionInstanceManager::find_instance(ReductionOpID& redop,
                                             Region region,
                                             FieldID field_id,
                                             Memory memory,
                                             Instance& result,
                                             const InstanceMappingPolicy& policy)
{
  auto finder = instance_sets_.find(FieldMemInfo(region.get_tree_id(), field_id, memory));
  return policy.allocation != AllocPolicy::MUST_ALLOC && finder != instance_sets_.end() &&
         finder->second.find_instance(redop, region, result, policy);
}

void ReductionInstanceManager::record_instance(ReductionOpID& redop,
                                               Region region,
                                               FieldID fid,
                                               Instance instance,
                                               const InstanceMappingPolicy& policy)
{
  const auto mem = instance.get_location();
  const auto tid = instance.get_tree_id();

  FieldMemInfo key(tid, fid, mem);
  auto finder = instance_sets_.find(key);
  if (finder != instance_sets_.end())
    instance_sets_[key].record_instance(redop, region, instance, policy);
  else {
    ReductionInstanceSet set;
    set.record_instance(redop, region, instance, policy);
    instance_sets_[key] = set;
  }
}

void ReductionInstanceManager::erase(Instance inst)
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

/*static*/ ReductionInstanceManager* ReductionInstanceManager::get_instance_manager()
{
  static ReductionInstanceManager* manager{nullptr};

  if (nullptr == manager) manager = new ReductionInstanceManager();
  return manager;
}

}  // namespace mapping
}  // namespace legate
