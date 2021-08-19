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

#include "mapping/instance_manager.h"
#include "utilities/dispatch.h"

namespace legate {
namespace mapping {

using namespace Legion;
using namespace Legion::Mapping;

bool InstanceInfoSet::has_instance(LogicalRegion region, PhysicalInstance& result) const
{
  auto finder = region_mapping.find(region);
  if (finder == region_mapping.end()) return false;
  const auto& info = instances[finder->second];
  result           = info->instance;
  return true;
}

uint32_t InstanceInfoSet::find_or_add_instance_info(PhysicalInstance inst,
                                                    LogicalRegion region,
                                                    const Domain& bound)
{
  uint32_t index = instances.size();
  for (uint32_t idx = 0; idx < instances.size(); idx++) {
    if (inst != instances[idx]->instance) continue;
    index = idx;
    break;
  }
  if (index == instances.size())
    instances.push_back(
      std::make_shared<InstanceInfo>(inst, bound, std::vector<LogicalRegion>({region})));
  region_mapping[region] = index;
  return index;
}

struct find_overlapping_instances_fn {
  template <int32_t DIM>
  Domain operator()(std::vector<uint32_t>& overlaps,
                    const Domain& domain,
                    const std::vector<std::shared_ptr<InstanceInfo>>& instances)
  {
    auto bound = domain.bounds<DIM, coord_t>();
    for (uint32_t idx = 0; idx < instances.size(); ++idx) {
      auto info       = instances[idx];
      Rect<DIM> other = info->bounding_box;
      auto intersect  = bound.intersection(other);
      if (intersect.empty()) continue;
      // Don't merge if the unused space would be more than the space saved
      auto union_bbox     = bound.union_bbox(other);
      size_t bound_volume = bound.volume();
      size_t union_volume = union_bbox.volume();
      // If it didn't get any bigger then we can keep going
      if (bound_volume == union_volume) continue;
      size_t intersect_volume = intersect.volume();
      // Only allow merging if it isn't "too big"
      // We define "too big" as the size of the "unused" points being bigger than the intersection
      if ((union_volume - (bound_volume + other.volume() - intersect_volume)) > intersect_volume)
        continue;
      overlaps.push_back(idx);
      bound = union_bbox;
    }
    return Domain(bound);
  }
};

Domain InstanceInfoSet::find_overlapping_instances(const Domain& domain,
                                                   std::vector<uint32_t>& overlaps) const
{
  return dim_dispatch(
    domain.get_dim(), find_overlapping_instances_fn{}, overlaps, domain, instances);
}

bool InstanceInfoSet::filter(PhysicalInstance inst)
{
  for (uint32_t idx = 0; idx < instances.size(); idx++)
    if (instances[idx]->instance == inst) {
      erase(idx);
      break;
    }
  return instances.empty();
}

void InstanceInfoSet::erase(uint32_t idx)
{
  // We also need to update any of the other region mappings
  for (auto it = region_mapping.begin(); it != region_mapping.end(); /*nothing*/) {
    if (it->second == idx) {
      auto to_delete = it++;
      region_mapping.erase(to_delete);
    } else {
      if (it->second > idx) it->second--;
      it++;
    }
  }
  instances.erase(instances.begin() + idx);
}

bool InstanceManager::find_instance(LogicalRegion region,
                                    FieldID field_id,
                                    Memory memory,
                                    PhysicalInstance& result)
{
  auto finder = instance_sets.find(FieldMemInfo(region.get_tree_id(), field_id, memory));
  return finder != instance_sets.end() && finder->second->has_instance(region, result);
}

std::shared_ptr<InstanceInfoSet> InstanceManager::find_instance_info_set(RegionTreeID tid,
                                                                         FieldID field_id,
                                                                         Memory memory)
{
  auto finder = instance_sets.find(FieldMemInfo(tid, field_id, memory));
  return finder == instance_sets.end() ? nullptr : finder->second;
}

std::shared_ptr<InstanceInfoSet> InstanceManager::find_or_create_instance_info_set(RegionTreeID tid,
                                                                                   FieldID field_id,
                                                                                   Memory memory)
{
  FieldMemInfo key(tid, field_id, memory);
  auto finder = instance_sets.find(key);
  if (finder != instance_sets.end())
    return finder->second;
  else {
    auto result        = std::make_shared<InstanceInfoSet>();
    instance_sets[key] = result;
    return result;
  }
}

void InstanceManager::filter(PhysicalInstance inst)
{
  const auto mem = inst.get_location();
  const auto tid = inst.get_tree_id();

  for (auto fit = instance_sets.begin(); fit != instance_sets.end(); /*nothing*/) {
    if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
      fit++;
      continue;
    }
    if (fit->second->filter(inst)) {
      auto to_delete = fit++;
      instance_sets.erase(to_delete);
    } else
      fit++;
  }
}

std::map<Legion::Memory, size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Legion::Memory, size_t> result;
  for (auto& pair : instance_sets) {
    auto& memory = pair.first.memory;
    if (result.find(memory) == result.end()) result[memory] = 0;
    for (auto& info : pair.second->instances) result[memory] += info->get_instance_size();
  }
  return result;
}

}  // namespace mapping
}  // namespace legate
