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

#include <memory>

#include "legion.h"

namespace legate {
namespace mapping {

struct FieldMemInfo {
 public:
  FieldMemInfo(Legion::RegionTreeID t, Legion::FieldID f, Legion::Memory m)
    : tid(t), fid(f), memory(m)
  {
  }

 public:
  inline bool operator==(const FieldMemInfo& rhs) const
  {
    return tid == rhs.tid && fid == rhs.fid && memory == rhs.memory;
  }

  inline bool operator<(const FieldMemInfo& rhs) const
  {
    if (tid < rhs.tid)
      return true;
    else if (tid > rhs.tid)
      return false;
    if (fid < rhs.fid)
      return true;
    else if (fid > rhs.fid)
      return false;
    return memory < rhs.memory;
  }

 public:
  Legion::RegionTreeID tid;
  Legion::FieldID fid;
  Legion::Memory memory;
};

struct InstanceInfo {
 public:
  InstanceInfo(Legion::Mapping::PhysicalInstance inst,
               const Legion::Domain& b,
               std::vector<Legion::LogicalRegion>&& rs)
    : instance(inst), bounding_box(b), regions(std::move(rs))
  {
    assert(bounding_box.get_dim() > 0);
  }

 public:
  size_t get_instance_size() const { return instance.get_instance_size(); }

 public:
  Legion::Mapping::PhysicalInstance instance;
  Legion::Domain bounding_box;
  std::vector<Legion::LogicalRegion> regions;
};

struct InstanceInfoSet {
 public:
  bool has_instance(Legion::LogicalRegion region, Legion::Mapping::PhysicalInstance& result) const;
  // This function should return std::set<std::shared_ptr<InstanceInfo>>
  Legion::Domain find_overlapping_instances(const Legion::Domain& domain,
                                            std::vector<uint32_t>& overlaps) const;

 public:
  uint32_t find_or_add_instance_info(Legion::Mapping::PhysicalInstance inst,
                                     Legion::LogicalRegion region,
                                     const Legion::Domain& bound);
  bool filter(Legion::Mapping::PhysicalInstance inst);
  void erase(uint32_t idx);

 public:
  // A list of instances that we have for this field in this memory
  std::vector<std::shared_ptr<InstanceInfo>> instances;
  // Mapping for logical regions that we already know have instances
  std::map<Legion::LogicalRegion, uint32_t> region_mapping;
  // void sanity_check(); ==> checks the round-trip property
  //    forall pair in instances.
  //      pair.first in pair.second->regions /\
  //      forall reg in pair.second->regions. instances[reg].second == pair.second
};

class InstanceManager {
 public:
  bool find_instance(Legion::LogicalRegion region,
                     Legion::FieldID field_id,
                     Legion::Memory memory,
                     Legion::Mapping::PhysicalInstance& result);
  std::shared_ptr<InstanceInfoSet> find_instance_info_set(Legion::RegionTreeID tid,
                                                          Legion::FieldID field_id,
                                                          Legion::Memory memory);
  std::shared_ptr<InstanceInfoSet> find_or_create_instance_info_set(Legion::RegionTreeID tid,
                                                                    Legion::FieldID field_id,
                                                                    Legion::Memory memory);
  void filter(Legion::Mapping::PhysicalInstance inst);
  std::map<Legion::Memory, size_t> aggregate_instance_sizes() const;

 public:
  std::map<FieldMemInfo, std::shared_ptr<InstanceInfoSet>> instance_sets;
};

}  // namespace mapping
}  // namespace legate
