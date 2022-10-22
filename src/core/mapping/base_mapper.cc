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

#include <cstdlib>
#include <sstream>
#include <unordered_map>

#include "legion/legion_mapping.h"
#include "mappers/mapping_utilities.h"

#include "core/data/store.h"
#include "core/mapping/base_mapper.h"
#include "core/mapping/instance_manager.h"
#include "core/mapping/operation.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/utilities/linearize.h"
#include "legate_defines.h"

using LegionTask = Legion::Task;
using LegionCopy = Legion::Copy;

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {
namespace mapping {

namespace {

const std::vector<StoreTarget>& default_store_targets(Processor::Kind kind)
{
  static const std::map<Processor::Kind, std::vector<StoreTarget>> defaults = {
    {Processor::LOC_PROC, {StoreTarget::SYSMEM}},
    {Processor::TOC_PROC, {StoreTarget::FBMEM, StoreTarget::ZCMEM}},
    {Processor::OMP_PROC, {StoreTarget::SOCKETMEM, StoreTarget::SYSMEM}},
  };

  auto finder = defaults.find(kind);
  if (defaults.end() == finder) LEGATE_ABORT;
  return finder->second;
}

}  // namespace

BaseMapper::BaseMapper(Runtime* rt, Machine m, const LibraryContext& ctx)
  : Mapper(rt->get_mapper_runtime()),
    legion_runtime(rt),
    machine(m),
    context(ctx),
    local_node(get_local_node()),
    total_nodes(get_total_nodes(m)),
    mapper_name(std::move(create_name(local_node))),
    logger(create_logger_name().c_str()),
    local_instances(InstanceManager::get_instance_manager())
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (auto local_proc : local_procs) {
    switch (local_proc.kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(local_proc);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(local_proc);
        break;
      }
      case Processor::OMP_PROC: {
        local_omps.push_back(local_proc);
        break;
      }
      default: break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (auto& local_gpu : local_gpus) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(local_gpu);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[local_gpu] = local_framebuffer.first();
  }
  for (auto& local_omp : local_omps) {
    Machine::MemoryQuery local_numa(machine);
    local_numa.local_address_space();
    local_numa.only_kind(Memory::SOCKET_MEM);
    local_numa.best_affinity_to(local_omp);
    if (local_numa.count() > 0)  // if we have NUMA memories then use them
      local_numa_domains[local_omp] = local_numa.first();
    else  // Otherwise we just use the local system memory
      local_numa_domains[local_omp] = local_system_memory;
  }
  generate_prime_factors();
}

BaseMapper::~BaseMapper(void)
{
  // Compute the size of all our remaining instances in each memory
  const char* show_usage = getenv("LEGATE_SHOW_USAGE");
  if (show_usage != nullptr) {
    auto mem_sizes             = local_instances->aggregate_instance_sizes();
    const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
      REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
    };
    for (auto& pair : mem_sizes) {
      const auto& mem       = pair.first;
      const size_t capacity = mem.capacity();
      logger.print(
        "%s used %ld bytes of %s memory %llx with "
        "%ld total bytes (%.2g%%)",
        context.get_library_name().c_str(),
        pair.second,
        memory_kinds[mem.kind()],
        mem.id,
        capacity,
        100.0 * double(pair.second) / capacity);
    }
  }
}

/*static*/ AddressSpace BaseMapper::get_local_node(void)
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

/*static*/ size_t BaseMapper::get_total_nodes(Machine m)
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<AddressSpace> spaces;
  for (auto proc : query) spaces.insert(proc.address_space());
  return spaces.size();
}

std::string BaseMapper::create_name(AddressSpace node) const
{
  std::stringstream ss;
  ss << context.get_library_name() << " on Node " << node;
  return ss.str();
}

std::string BaseMapper::create_logger_name() const
{
  std::stringstream ss;
  ss << context.get_library_name() << ".mapper";
  return ss.str();
}

const char* BaseMapper::get_mapper_name(void) const { return mapper_name.c_str(); }

Mapper::MapperSyncModel BaseMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void BaseMapper::select_task_options(const MapperContext ctx,
                                     const LegionTask& task,
                                     TaskOptions& output)
{
  std::vector<TaskTarget> options;
  if (!local_gpus.empty() && has_variant(ctx, task, Processor::TOC_PROC))
    options.push_back(TaskTarget::GPU);
  if (!local_omps.empty() && has_variant(ctx, task, Processor::OMP_PROC))
    options.push_back(TaskTarget::OMP);
  options.push_back(TaskTarget::CPU);

  Task legate_task(&task, context, runtime, ctx);
  auto target = task_target(legate_task, options);

  dispatch(target, [&output](auto& procs) { output.initial_proc = procs.front(); });
  // We never want valid instances
  output.valid_instances = false;
}

void BaseMapper::premap_task(const MapperContext ctx,
                             const LegionTask& task,
                             const PremapTaskInput& input,
                             PremapTaskOutput& output)
{
  // NO-op since we know that all our futures should be mapped in the system memory
}

void BaseMapper::slice_auto_task(const MapperContext ctx,
                                 const LegionTask& task,
                                 const SliceTaskInput& input,
                                 SliceTaskOutput& output)
{
  LegateProjectionFunctor* key_functor = nullptr;
  for (auto& req : task.regions)
    if (req.tag == LEGATE_CORE_KEY_STORE_TAG) {
      key_functor = find_legate_projection_functor(req.projection);
      break;
    }

  // For multi-node cases we should already have been sharded so we
  // should just have one or a few points here on this node, so iterate
  // them and round-robin them across the local processors here
  output.slices.reserve(input.domain.get_volume());

  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  auto round_robin = [&](auto& procs) {
    if (nullptr != key_functor) {
      auto lo = key_functor->project_point(sharding_domain.lo(), sharding_domain);
      auto hi = key_functor->project_point(sharding_domain.hi(), sharding_domain);
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        auto p   = key_functor->project_point(itr.p, sharding_domain);
        auto idx = linearize(lo, hi, p);
        output.slices.push_back(TaskSlice(
          Domain(itr.p, itr.p), procs[idx % procs.size()], false /*recurse*/, false /*stealable*/));
      }
    } else {
      auto lo = sharding_domain.lo();
      auto hi = sharding_domain.hi();
      for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        auto idx = linearize(lo, hi, itr.p);
        output.slices.push_back(TaskSlice(
          Domain(itr.p, itr.p), procs[idx % procs.size()], false /*recurse*/, false /*stealable*/));
      }
    }
  };

  dispatch(task.target_proc.kind(), round_robin);
}

void BaseMapper::generate_prime_factor(const std::vector<Processor>& processors,
                                       Processor::Kind kind)
{
  std::vector<int32_t>& factors = all_factors[kind];
  int32_t num_procs             = static_cast<int32_t>(processors.size());

  auto generate_factors = [&](int32_t factor) {
    while (num_procs % factor == 0) {
      factors.push_back(factor);
      num_procs /= factor;
    }
  };
  generate_factors(2);
  generate_factors(3);
  generate_factors(5);
  generate_factors(7);
  generate_factors(11);
}

void BaseMapper::generate_prime_factors()
{
  if (local_gpus.size() > 0) generate_prime_factor(local_gpus, Processor::TOC_PROC);
  if (local_omps.size() > 0) generate_prime_factor(local_omps, Processor::OMP_PROC);
  if (local_cpus.size() > 0) generate_prime_factor(local_cpus, Processor::LOC_PROC);
}

const std::vector<int32_t> BaseMapper::get_processor_grid(Legion::Processor::Kind kind,
                                                          int32_t ndim)
{
  auto key    = std::make_pair(kind, ndim);
  auto finder = proc_grids.find(key);
  if (finder != proc_grids.end()) return finder->second;

  int32_t num_procs = dispatch(kind, [](auto& procs) { return procs.size(); });

  std::vector<int32_t> grid;
  auto factor_it = all_factors[kind].begin();
  grid.resize(ndim, 1);

  while (num_procs > 1) {
    auto min_it = std::min_element(grid.begin(), grid.end());
    auto factor = *factor_it++;
    (*min_it) *= factor;
    num_procs /= factor;
  }

  auto& pitches = proc_grids[key];
  pitches.resize(ndim, 1);
  for (int32_t dim = 1; dim < ndim; ++dim) pitches[dim] = pitches[dim - 1] * grid[dim - 1];

  return pitches;
}

void BaseMapper::slice_manual_task(const MapperContext ctx,
                                   const LegionTask& task,
                                   const SliceTaskInput& input,
                                   SliceTaskOutput& output)
{
  output.slices.reserve(input.domain.get_volume());

  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  auto distribute = [&](auto& procs) {
    auto ndim       = input.domain.dim;
    auto& proc_grid = get_processor_grid(task.target_proc.kind(), ndim);
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
      int32_t idx = 0;
      for (int32_t dim = 0; dim < ndim; ++dim) idx += proc_grid[dim] * itr.p[dim];
      output.slices.push_back(TaskSlice(
        Domain(itr.p, itr.p), procs[idx % procs.size()], false /*recurse*/, false /*stealable*/));
    }
  };

  dispatch(task.target_proc.kind(), distribute);
}

void BaseMapper::slice_round_robin_task(const MapperContext ctx,
                                        const LegionTask& task,
                                        const SliceTaskInput& input,
                                        SliceTaskOutput& output)
{
  // If we're here, that means that the task has no region that we can key off
  // to distribute them reasonably. In this case, we just do a round-robin
  // assignment.

  output.slices.reserve(input.domain.get_volume());

  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  auto distribute = [&](auto& procs) {
    size_t idx = 0;
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
      output.slices.push_back(TaskSlice(
        Domain(itr.p, itr.p), procs[idx++ % procs.size()], false /*recurse*/, false /*stealable*/));
    }
  };

  dispatch(task.target_proc.kind(), distribute);
}

void BaseMapper::slice_task(const MapperContext ctx,
                            const LegionTask& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  if (task.tag == LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG) {
    if (task.regions.size() == 0)
      slice_round_robin_task(ctx, task, input, output);
    else
      slice_manual_task(ctx, task, input, output);
  } else
    slice_auto_task(ctx, task, input, output);
}

bool BaseMapper::has_variant(const MapperContext ctx, const LegionTask& task, Processor::Kind kind)
{
  const std::pair<TaskID, Processor::Kind> key(task.task_id, kind);
  // Check to see if we already have it
  auto finder = leaf_variants.find(key);
  if ((finder != leaf_variants.end()) && (finder->second != 0)) return true;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  // Process all the results, record if we found what we were looking for
  bool has_leaf = false;
  for (auto vid : variants) {
    assert(vid > 0);
    switch (vid) {
      case LEGATE_CPU_VARIANT:
      case LEGATE_OMP_VARIANT:
      case LEGATE_GPU_VARIANT: {
        has_leaf           = true;
        leaf_variants[key] = vid;
        break;
      }
      default:         // TODO: handle vectorized variants
        LEGATE_ABORT;  // unhandled variant kind
    }
  }
  if (!has_leaf) leaf_variants[key] = 0;
  return has_leaf;
}

VariantID BaseMapper::find_variant(const MapperContext ctx,
                                   const LegionTask& task,
                                   Processor::Kind kind)
{
  const std::pair<TaskID, Processor::Kind> key(task.task_id, kind);
  auto finder = leaf_variants.find(key);
  if ((finder != leaf_variants.end()) && (finder->second != 0)) return finder->second;
  // Haven't seen it before so let's look it up to make sure it exists
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, key.first, variants, key.second);
  VariantID result = 0;  // 0 is reserved
  bool has_leaf    = false;
  // Process all the results, record if we found what we were looking for
  for (auto vid : variants) {
    assert(vid > 0);
    switch (vid) {
      case LEGATE_CPU_VARIANT:
      case LEGATE_OMP_VARIANT:
      case LEGATE_GPU_VARIANT: {
        has_leaf           = true;
        leaf_variants[key] = vid;
        result             = vid;
        break;
      }
      default:         // TODO: handle vectorized variants
        LEGATE_ABORT;  // unhandled variant kind
    }
  }
  if (!has_leaf) leaf_variants[key] = 0;
  // We must always be able to find the variant;
  assert(result != 0);
  return result;
}

void BaseMapper::map_task(const MapperContext ctx,
                          const LegionTask& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output)
{
#ifdef DEBUG_LEGATE
  logger.debug() << "Entering map_task for " << Utilities::to_string(runtime, ctx, task);
#endif

  // Should never be mapping the top-level task here
  assert(task.get_depth() > 0);

  // Let's populate easy outputs first
  output.chosen_variant = find_variant(ctx, task, task.target_proc.kind());
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);

  Task legate_task(&task, context, runtime, ctx);

  const auto& options = default_store_targets(task.target_proc.kind());

  output.chosen_instances.resize(task.regions.size());
  std::map<const RegionRequirement*, std::vector<PhysicalInstance>*> output_map;
  for (uint32_t idx = 0; idx < task.regions.size(); ++idx)
    output_map[&task.regions[idx]] = &output.chosen_instances[idx];

  auto mappings = store_mappings(legate_task, options);

  std::map<RegionField::Id, uint32_t> client_mapped_regions;
  std::map<uint32_t, uint32_t> client_mapped_futures;
  for (uint32_t mapping_idx = 0; mapping_idx < mappings.size(); ++mapping_idx) {
    auto& mapping = mappings[mapping_idx];

    assert(mapping.stores.size() > 0);
    for (uint32_t store_idx = 1; store_idx < mapping.stores.size(); ++store_idx) {
      if (!mapping.stores[store_idx].can_colocate_with(mapping.stores[0])) {
        logger.error("Mapper %s tried to colocate stores that cannot colocate", get_mapper_name());
        LEGATE_ABORT;
      }
    }

    if (mapping.stores.size() > 1 && mapping.policy.ordering.relative) {
      logger.error("Colocation with relative dimension ordering is illegal");
      LEGATE_ABORT;
    }

    for (auto& store : mapping.stores) {
      if (store.is_future()) {
        auto fut_idx                   = store.future().index();
        client_mapped_futures[fut_idx] = mapping_idx;
        continue;
      }

      auto& rf = store.region_field();
      auto key = rf.unique_id();

      auto finder = client_mapped_regions.find(key);
      // If this is the first store mapping for this requirement,
      // we record the mapping index for future reference.
      if (finder == client_mapped_regions.end()) client_mapped_regions[key] = mapping_idx;
      // If we're still in the same store mapping, we know for sure
      // that the mapping is consistent.
      else {
        if (finder->second == mapping_idx) continue;
        // Otherwise, we do consistency checking
        auto& other_mapping = mappings[finder->second];
        if (mapping.policy != other_mapping.policy) {
          logger.error("Mapper %s returned inconsistent store mappings", get_mapper_name());
          LEGATE_ABORT;
        }
      }
    }
  }

  // Generate default mappings for stores that are not yet mapped by the client mapper
  auto default_option            = options.front();
  auto generate_default_mappings = [&](auto& stores, bool exact) {
    for (auto& store : stores) {
      if (store.is_future()) {
        auto fut_idx = store.future().index();
        if (client_mapped_futures.find(fut_idx) == client_mapped_futures.end())
          mappings.push_back(StoreMapping::default_mapping(store, default_option, exact));
        continue;
      } else {
        auto key = store.region_field().unique_id();
        if (client_mapped_regions.find(key) != client_mapped_regions.end()) continue;
        client_mapped_regions[key] = static_cast<int32_t>(mappings.size());
        mappings.push_back(StoreMapping::default_mapping(store, default_option, exact));
      }
    }
  };

  generate_default_mappings(legate_task.inputs(), false);
  generate_default_mappings(legate_task.outputs(), false);
  generate_default_mappings(legate_task.reductions(), false);

  bool can_fail = true;
  std::map<PhysicalInstance, std::set<int32_t>> instance_to_mappings;
  std::map<int32_t, PhysicalInstance> mapping_to_instance;
  std::vector<bool> handled(mappings.size(), false);

  // See case of failed instance creation below
  auto tighten_write_reqs = [&]() {
    for (int32_t mapping_idx = 0; mapping_idx < mappings.size(); ++mapping_idx) {
      auto& mapping      = mappings[mapping_idx];
      PrivilegeMode priv = LEGION_NO_ACCESS;
      for (auto* req : mapping.requirements()) priv |= req->privilege;
      if (!(priv & LEGION_WRITE_PRIV) || mapping.policy.exact) continue;

#ifdef DEBUG_LEGATE
      std::stringstream reqs_ss;
      for (auto req_idx : mapping.requirement_indices()) reqs_ss << " " << req_idx;
      logger.debug() << "Task " << task.get_unique_id()
                     << ": tightened mapping policy for reqs:" << reqs_ss.str();
#endif

      mapping.policy.exact = true;
      if (!handled[mapping_idx]) continue;
      handled[mapping_idx] = false;
      auto m2i_it          = mapping_to_instance.find(mapping_idx);
      if (m2i_it == mapping_to_instance.end()) continue;
      PhysicalInstance inst = m2i_it->second;
      mapping_to_instance.erase(m2i_it);
      auto i2m_it = instance_to_mappings.find(inst);
      i2m_it->second.erase(mapping_idx);
      if (i2m_it->second.empty()) {
        runtime->release_instance(ctx, inst);
        instance_to_mappings.erase(i2m_it);
      }
    }
  };

  // Mapping each field separately for each of the logical regions
  for (int32_t mapping_idx = 0; mapping_idx < mappings.size(); ++mapping_idx) {
    if (handled[mapping_idx]) continue;
    auto& mapping = mappings[mapping_idx];

    if (mapping.for_unbound_stores()) {
      auto req_indices = mapping.requirement_indices();
      for (auto req_idx : req_indices) {
        output.output_targets[req_idx] = get_target_memory(task.target_proc, mapping.policy.target);
        auto ndim                      = mapping.stores.front().dim();
        // FIXME: Unbound stores can have more than one dimension later
        std::vector<DimensionKind> dimension_ordering;
        for (int32_t dim = ndim - 1; dim >= 0; --dim)
          dimension_ordering.push_back(
            static_cast<DimensionKind>(static_cast<int32_t>(DimensionKind::LEGION_DIM_X) + dim));
        dimension_ordering.push_back(DimensionKind::LEGION_DIM_F);
        output.output_constraints[req_idx].ordering_constraint =
          OrderingConstraint(dimension_ordering, false);
      }
      handled[mapping_idx] = true;
      continue;
    }

    auto reqs = mapping.requirements();
    if (reqs.empty()) {
      // This is a mapping for futures
      StoreTarget target = mapping.policy.target;
#ifdef LEGATE_NO_FUTURES_ON_FB
      if (target == StoreTarget::FBMEM) target = StoreTarget::ZCMEM;
#endif
      output.future_locations.push_back(get_target_memory(task.target_proc, target));
      handled[mapping_idx] = true;
      continue;
    }

#ifdef DEBUG_LEGATE
    std::stringstream reqs_ss;
    for (auto req_idx : mapping.requirement_indices()) reqs_ss << " " << req_idx;
#endif

    // Get an instance and acquire it if necessary. If the acquire fails then prune it from the
    // mapper's data structures and retry, until we succeed or map_legate_store fails with an out of
    // memory error.
    PhysicalInstance result;
    while (map_legate_store(ctx, task, mapping, reqs, task.target_proc, result, can_fail)) {
      if (result == PhysicalInstance()) break;
      if (instance_to_mappings.count(result) > 0 || runtime->acquire_instance(ctx, result)) {
#ifdef DEBUG_LEGATE
        logger.debug() << "Task " << task.get_unique_id() << ": acquired instance " << result
                       << " for reqs:" << reqs_ss.str();
#endif
        break;
      }
#ifdef DEBUG_LEGATE
      logger.debug() << "Task " << task.get_unique_id() << ": failed to acquire instance " << result
                     << " for reqs:" << reqs_ss.str();
#endif
      AutoLock lock(ctx, local_instances->manager_lock());
      local_instances->erase(result);
    }

    // If instance creation failed we try mapping all stores again, but request tight instances for
    // write requirements. The hope is that these write requirements cover the entire region (i.e.
    // they use a complete partition), so the new tight instances will invalidate any pre-existing
    // "bloated" instances for the same region, freeing up enough memory so that mapping can succeed
    if (result == PhysicalInstance()) {
#ifdef DEBUG_LEGATE
      logger.debug() << "Task " << task.get_unique_id()
                     << ": failed mapping for reqs:" << reqs_ss.str();
#endif
      assert(can_fail);
      tighten_write_reqs();
      mapping_idx = -1;
      can_fail    = false;
      continue;
    }

    // Success; record the instance for this mapping.
#ifdef DEBUG_LEGATE
    logger.debug() << "Task " << task.get_unique_id()
                   << ": completed mapping for reqs:" << reqs_ss.str();
#endif
    instance_to_mappings[result].insert(mapping_idx);
    mapping_to_instance[mapping_idx] = result;
    handled[mapping_idx]             = true;
  }

  // Succeeded in mapping all stores, record it on map_task output.
  for (const auto& m2i : mapping_to_instance)
    for (auto req : mappings[m2i.first].requirements())
      if (req->region.exists()) output_map[req]->push_back(m2i.second);
}

void BaseMapper::map_replicate_task(const MapperContext ctx,
                                    const LegionTask& task,
                                    const MapTaskInput& input,
                                    const MapTaskOutput& def_output,
                                    MapReplicateTaskOutput& output)
{
  LEGATE_ABORT;
}

bool BaseMapper::find_existing_instance(const MapperContext ctx,
                                        LogicalRegion region,
                                        FieldID fid,
                                        Memory target_memory,
                                        PhysicalInstance& result,
                                        Strictness strictness,
                                        bool acquire_instance_lock)
{
  std::unique_ptr<AutoLock> lock =
    acquire_instance_lock ? std::make_unique<AutoLock>(ctx, local_instances->manager_lock())
                          : nullptr;
  // See if we already have it in our local instances
  if (local_instances->find_instance(region, fid, target_memory, result))
    return true;
  else if (strictness == Strictness::strict)
    return false;

  // See if we can find an existing instance in any memory
  if (local_instances->find_instance(region, fid, local_system_memory, result)) return true;

  for (auto& pair : local_frame_buffers)
    if (local_instances->find_instance(region, fid, pair.second, result)) return true;

  for (auto& pair : local_numa_domains)
    if (local_instances->find_instance(region, fid, pair.second, result)) return true;

  return false;
}

Memory BaseMapper::get_target_memory(Processor proc, StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return local_system_memory;
    case StoreTarget::FBMEM: return local_frame_buffers[proc];
    case StoreTarget::ZCMEM: return local_zerocopy_memory;
    case StoreTarget::SOCKETMEM: return local_numa_domains[proc];
    default: LEGATE_ABORT;
  }
  assert(false);
  return Memory::NO_MEMORY;
}

bool BaseMapper::map_legate_store(const MapperContext ctx,
                                  const Mappable& mappable,
                                  const StoreMapping& mapping,
                                  const std::set<const RegionRequirement*>& reqs,
                                  Processor target_proc,
                                  PhysicalInstance& result,
                                  bool can_fail)
{
  const auto& policy = mapping.policy;
  std::vector<LogicalRegion> regions;
  for (auto* req : reqs) regions.push_back(req->region);
  auto target_memory = get_target_memory(target_proc, policy.target);

  ReductionOpID redop = 0;
  bool first          = true;
  for (auto* req : reqs) {
    if (first)
      redop = req->redop;
    else {
      if (redop != req->redop) {
        logger.error(
          "Colocated stores should be either non-reduction arguments "
          "or reductions with the same reduction operator.");
        LEGATE_ABORT;
      }
    }
  }

  // Generate layout constraints from the store mapping
  LayoutConstraintSet layout_constraints;
  mapping.populate_layout_constraints(layout_constraints);

  // If we're making a reduction instance, we should just make it now
  if (redop != 0) {
    layout_constraints.add_constraint(SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, redop));
    size_t footprint = 0;
    if (runtime->create_physical_instance(ctx,
                                          target_memory,
                                          layout_constraints,
                                          regions,
                                          result,
                                          true /*acquire*/,
                                          LEGION_GC_DEFAULT_PRIORITY,
                                          false /*tight bounds*/,
                                          &footprint)) {
#ifdef DEBUG_LEGATE
      Realm::LoggerMessage msg = logger.debug();
      msg << "Operation " << mappable.get_unique_id() << ": created reduction instance " << result
          << " for";
      for (LogicalRegion r : regions) msg << " " << r;
      msg << " (size: " << footprint << " bytes, memory: " << target_memory << ")";
#endif
      // We already did the acquire
      return false;
    }
    if (!can_fail)
      report_failed_mapping(mappable, mapping.requirement_index(), target_memory, redop);
    return true;
  }

  auto& fields = layout_constraints.field_constraint.field_set;

  // We need to hold the instance manager lock as we're about to try to find an instance
  AutoLock lock(ctx, local_instances->manager_lock());

  // This whole process has to appear atomic
  runtime->disable_reentrant(ctx);

  // See if we already have it in our local instances
  if (fields.size() == 1 && regions.size() == 1 &&
      local_instances->find_instance(
        regions.front(), fields.front(), target_memory, result, policy)) {
#ifdef DEBUG_LEGATE
    logger.debug() << "Operation " << mappable.get_unique_id() << ": reused cached instance "
                   << result << " for " << regions.front();
#endif
    runtime->enable_reentrant(ctx);
    // Needs acquire to keep the runtime happy
    return true;
  }

  std::shared_ptr<RegionGroup> group{nullptr};

  // Haven't made this instance before, so make it now
  if (fields.size() == 1 && regions.size() == 1) {
    // When the client mapper didn't request colocation and also didn't want the instance
    // to be exact, we can do an interesting optimization here to try to reduce unnecessary
    // inter-memory copies. For logical regions that are overlapping we try
    // to accumulate as many as possible into one physical instance and use
    // that instance for all the tasks for the different regions.
    // First we have to see if there is anything we overlap with
    auto fid            = fields.front();
    const IndexSpace is = regions.front().get_index_space();
    const Domain domain = runtime->get_index_space_domain(ctx, is);
    group =
      local_instances->find_region_group(regions.front(), domain, fid, target_memory, policy.exact);
    regions = group->get_regions();
  }

  bool created     = false;
  bool success     = false;
  size_t footprint = 0;

  switch (policy.allocation) {
    case AllocPolicy::MAY_ALLOC: {
      success = runtime->find_or_create_physical_instance(ctx,
                                                          target_memory,
                                                          layout_constraints,
                                                          regions,
                                                          result,
                                                          created,
                                                          true /*acquire*/,
                                                          LEGION_GC_DEFAULT_PRIORITY,
                                                          policy.exact /*tight bounds*/,
                                                          &footprint);
      break;
    }
    case AllocPolicy::MUST_ALLOC: {
      success = runtime->create_physical_instance(ctx,
                                                  target_memory,
                                                  layout_constraints,
                                                  regions,
                                                  result,
                                                  true /*acquire*/,
                                                  LEGION_GC_DEFAULT_PRIORITY,
                                                  policy.exact /*tight bounds*/,
                                                  &footprint);
      break;
    }
    default: LEGATE_ABORT;  // should never get here
  }

  if (success) {
    // We succeeded in making the instance where we want it
    assert(result.exists());
#ifdef DEBUG_LEGATE
    if (created) {
      logger.debug() << "Operation " << mappable.get_unique_id() << ": created instance " << result
                     << " for " << *group << " (size: " << footprint
                     << " bytes, memory: " << target_memory << ")";
    } else {
      logger.debug() << "Operation " << mappable.get_unique_id() << ": found instance " << result
                     << " for " << *group;
    }
#endif
    // Only save the result for future use if it is not an external instance
    if (!result.is_external_instance() && group != nullptr) {
      assert(fields.size() == 1);
      auto fid = fields.front();
      local_instances->record_instance(group, fid, result, policy);
    }
    runtime->enable_reentrant(ctx);
    // We made it so no need for an acquire
    return false;
  }
  // Done with the atomic part
  runtime->enable_reentrant(ctx);

  // If we make it here then we failed entirely
  if (!can_fail) {
    auto req_indices = mapping.requirement_indices();
    for (auto req_idx : req_indices) report_failed_mapping(mappable, req_idx, target_memory, redop);
  }
  return true;
}

void BaseMapper::filter_failed_acquires(const MapperContext ctx,
                                        std::vector<PhysicalInstance>& needed_acquires,
                                        std::set<PhysicalInstance>& failed_acquires)
{
  AutoLock lock(ctx, local_instances->manager_lock());
  for (auto& instance : needed_acquires) {
    if (failed_acquires.find(instance) != failed_acquires.end()) continue;
    failed_acquires.insert(instance);
    local_instances->erase(instance);
  }
  needed_acquires.clear();
}

void BaseMapper::report_failed_mapping(const Mappable& mappable,
                                       uint32_t index,
                                       Memory target_memory,
                                       ReductionOpID redop)
{
  const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
    REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
  };
  std::string provenance = mappable.get_provenance_string();
  if (provenance.empty()) provenance = "unknown provenance";
  switch (mappable.get_mappable_type()) {
    case Mappable::TASK_MAPPABLE: {
      const auto task = mappable.as_task();
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of task %s [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          task->get_task_name(),
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "task %s [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          task->get_task_name(),
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::COPY_MAPPABLE: {
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of copy [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "copy [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::INLINE_MAPPABLE: {
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of inline mapping [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "inline mapping [%s] (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          provenance.c_str(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::PARTITION_MAPPABLE: {
      assert(redop == 0);
      logger.error(
        "Mapper %s failed to map region requirement %d of "
        "partition (UID %lld) into %s memory " IDFMT,
        get_mapper_name(),
        index,
        mappable.get_unique_id(),
        memory_kinds[target_memory.kind()],
        target_memory.id);
      break;
    }
    default: LEGATE_ABORT;  // should never get here
  }
  LEGATE_ABORT;
}

void BaseMapper::select_task_variant(const MapperContext ctx,
                                     const LegionTask& task,
                                     const SelectVariantInput& input,
                                     SelectVariantOutput& output)
{
  output.chosen_variant = find_variant(ctx, task, input.processor.kind());
}

void BaseMapper::postmap_task(const MapperContext ctx,
                              const LegionTask& task,
                              const PostMapInput& input,
                              PostMapOutput& output)
{
  // We should currently never get this call in Legate
  LEGATE_ABORT;
}

void BaseMapper::select_task_sources(const MapperContext ctx,
                                     const LegionTask& task,
                                     const SelectTaskSrcInput& input,
                                     SelectTaskSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::legate_select_sources(const MapperContext ctx,
                                       const PhysicalInstance& target,
                                       const std::vector<PhysicalInstance>& sources,
                                       std::deque<PhysicalInstance>& ranking)
{
  std::map<Memory, uint32_t /*bandwidth*/> source_memories;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination, we'll only rank sources from the
  // local node if there are any
  bool all_local = false;
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory destination_memory = target.get_location();
  std::vector<MemoryMemoryAffinity> affinity(1);
  // fill in a vector of the sources with their bandwidths and sort them
  std::vector<std::pair<PhysicalInstance, uint32_t /*bandwidth*/>> band_ranking;
  for (uint32_t idx = 0; idx < sources.size(); idx++) {
    const PhysicalInstance& instance = sources[idx];
    Memory location                  = instance.get_location();
    if (location.address_space() == local_node) {
      if (!all_local) {
        source_memories.clear();
        band_ranking.clear();
        all_local = true;
      }
    } else if (all_local)  // Skip any remote instances once we're local
      continue;
    auto finder = source_memories.find(location);
    if (finder == source_memories.end()) {
      affinity.clear();
      machine.get_mem_mem_affinity(
        affinity, location, destination_memory, false /*not just local affinities*/);
      uint32_t memory_bandwidth = 0;
      if (!affinity.empty()) {
        assert(affinity.size() == 1);
        memory_bandwidth = affinity[0].bandwidth;
#if 0
          } else {
            // TODO: More graceful way of dealing with multi-hop copies
            logger.warning("Legate mapper is potentially "
                              "requesting a multi-hop copy between memories "
                              IDFMT " and " IDFMT "!", location.id,
                              destination_memory.id);
#endif
      }
      source_memories[location] = memory_bandwidth;
      band_ranking.push_back(std::pair<PhysicalInstance, uint32_t>(instance, memory_bandwidth));
    } else
      band_ranking.push_back(std::pair<PhysicalInstance, uint32_t>(instance, finder->second));
  }
  assert(!band_ranking.empty());
  // Easy case of only one instance
  if (band_ranking.size() == 1) {
    ranking.push_back(band_ranking.begin()->first);
    return;
  }
  // Sort them by bandwidth
  std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
  // Iterate from largest bandwidth to smallest
  for (auto it = band_ranking.rbegin(); it != band_ranking.rend(); ++it)
    ranking.push_back(it->first);
}

void BaseMapper::speculate(const MapperContext ctx,
                           const LegionTask& task,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const LegionTask& task,
                                  const TaskProfilingInfo& input)
{
  // Shouldn't get any profiling feedback currently
  LEGATE_ABORT;
}

ShardingID BaseMapper::find_sharding_functor_by_key_store_projection(
  const std::vector<RegionRequirement>& requirements)
{
  ProjectionID proj_id = 0;
  for (auto& requirement : requirements)
    if (LEGATE_CORE_KEY_STORE_TAG == requirement.tag) {
      proj_id = requirement.projection;
      break;
    }
  return find_sharding_functor_by_projection_functor(proj_id);
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const LegionTask& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = task.is_index_space
                            ? find_sharding_functor_by_key_store_projection(task.regions)
                            : find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::map_inline(const MapperContext ctx,
                            const InlineMapping& inline_op,
                            const MapInlineInput& input,
                            MapInlineOutput& output)
{
  Processor target_proc{Processor::NO_PROC};
  if (!local_omps.empty())
    target_proc = local_omps.front();
  else
    target_proc = local_cpus.front();

  auto store_target = default_store_targets(target_proc.kind()).front();

#ifdef DEBUG_LEGATE
  assert(inline_op.requirement.instance_fields.size() == 1);
#endif

  Store store(legion_runtime->get_mapper_runtime(), ctx, &inline_op.requirement);
  StoreMapping mapping = StoreMapping::default_mapping(store, store_target, false);

  auto reqs = mapping.requirements();
  output.chosen_instances.resize(1);
  PhysicalInstance& result = output.chosen_instances.front();
  bool can_fail            = false;
  while (map_legate_store(ctx, inline_op, mapping, reqs, target_proc, result, can_fail)) {
    if (result == PhysicalInstance()) break;
    if (runtime->acquire_instance(ctx, result)) {
#ifdef DEBUG_LEGATE
      logger.debug() << "Inline mapping " << inline_op.get_unique_id() << ": acquired instance "
                     << result << " for reqs: 0";
#endif
      break;
    }
#ifdef DEBUG_LEGATE
    logger.debug() << "Inline mapping " << inline_op.get_unique_id()
                   << ": failed to acquire instance " << result << " for reqs: 0";
#endif
    AutoLock lock(ctx, local_instances->manager_lock());
    local_instances->erase(result);
  }
}

void BaseMapper::select_inline_sources(const MapperContext ctx,
                                       const InlineMapping& inline_op,
                                       const SelectInlineSrcInput& input,
                                       SelectInlineSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const InlineMapping& inline_op,
                                  const InlineProfilingInfo& input)
{
  // No profiling yet for inline mappings
  LEGATE_ABORT;
}

void BaseMapper::map_copy(const MapperContext ctx,
                          const LegionCopy& copy,
                          const MapCopyInput& input,
                          MapCopyOutput& output)
{
  Processor target_proc{Processor::NO_PROC};

  uint32_t proc_id = 0;
  if (copy.is_index_space) {
    Domain sharding_domain = copy.index_domain;
    if (copy.sharding_space.exists())
      sharding_domain = runtime->get_index_space_domain(ctx, copy.sharding_space);

    // FIXME: We might later have non-identity projections for copy requirements,
    // in which case we should find the key store and use its projection functor
    // for the linearization
    auto* key_functor = find_legate_projection_functor(0);

    if (key_functor != nullptr) {
      auto lo = key_functor->project_point(sharding_domain.lo(), sharding_domain);
      auto hi = key_functor->project_point(sharding_domain.hi(), sharding_domain);
      auto p  = key_functor->project_point(copy.index_point, sharding_domain);
      proc_id = linearize(lo, hi, p);
    } else {
      proc_id = linearize(sharding_domain.lo(), sharding_domain.hi(), copy.index_point);
    }
  }
  if (!local_gpus.empty())
    target_proc = local_gpus[proc_id % local_gpus.size()];
  else if (!local_omps.empty())
    target_proc = local_omps[proc_id % local_omps.size()];
  else
    target_proc = local_cpus[proc_id % local_cpus.size()];

  auto store_target = default_store_targets(target_proc.kind()).front();

  Copy legate_copy(&copy, runtime, ctx);

  std::map<const RegionRequirement*, std::vector<PhysicalInstance>*> output_map;
  auto add_to_output_map = [&output_map](auto& reqs, auto& instances) {
    instances.resize(reqs.size());
    for (uint32_t idx = 0; idx < reqs.size(); ++idx) output_map[&reqs[idx]] = &instances[idx];
  };
  add_to_output_map(copy.src_requirements, output.src_instances);
  add_to_output_map(copy.dst_requirements, output.dst_instances);

#ifdef DEBUG_LEGATE
  assert(copy.src_indirect_requirements.size() <= 1);
  assert(copy.dst_indirect_requirements.size() <= 1);
#endif
  if (!copy.src_indirect_requirements.empty()) {
    // This is to make the push_back call later add the isntance to the right place
    output.src_indirect_instances.clear();
    output_map[&copy.src_indirect_requirements.front()] = &output.src_indirect_instances;
  }
  if (!copy.dst_indirect_requirements.empty()) {
    // This is to make the push_back call later add the isntance to the right place
    output.dst_indirect_instances.clear();
    output_map[&copy.dst_indirect_requirements.front()] = &output.dst_indirect_instances;
  }

  std::vector<StoreMapping> mappings;

  for (auto& store : legate_copy.inputs())
    mappings.push_back(StoreMapping::default_mapping(store, store_target, false));
  for (auto& store : legate_copy.outputs())
    mappings.push_back(StoreMapping::default_mapping(store, store_target, false));
  for (auto& store : legate_copy.input_indirections())
    mappings.push_back(StoreMapping::default_mapping(store, store_target, false));
  for (auto& store : legate_copy.output_indirections())
    mappings.push_back(StoreMapping::default_mapping(store, store_target, false));

  bool can_fail = false;

  std::map<int32_t, PhysicalInstance> mapping_to_instance;
  for (int32_t mapping_idx = 0; mapping_idx < mappings.size(); ++mapping_idx) {
    auto& mapping = mappings[mapping_idx];
    auto reqs     = mapping.requirements();
#ifdef DEBUG_LEGATE
    std::stringstream reqs_ss;
    for (auto req_idx : mapping.requirement_indices()) reqs_ss << " " << req_idx;
#endif

    PhysicalInstance result;
    while (map_legate_store(ctx, copy, mapping, reqs, target_proc, result, can_fail)) {
      if (result == PhysicalInstance()) break;
      if (runtime->acquire_instance(ctx, result)) {
#ifdef DEBUG_LEGATE
        logger.debug() << "Copy " << copy.get_unique_id() << ": acquired instance " << result
                       << " for reqs:" << reqs_ss.str();
#endif
        break;
      }
#ifdef DEBUG_LEGATE
      logger.debug() << "Copy " << copy.get_unique_id() << ": failed to acquire instance " << result
                     << " for reqs:" << reqs_ss.str();
#endif
      AutoLock lock(ctx, local_instances->manager_lock());
      local_instances->erase(result);
    }
    mapping_to_instance[mapping_idx] = result;
  }

  // Succeeded in mapping all stores, record it on map_copy output.
  for (const auto& m2i : mapping_to_instance)
    for (auto req : mappings[m2i.first].requirements())
      if (req->region.exists()) output_map[req]->push_back(m2i.second);
}

void BaseMapper::select_copy_sources(const MapperContext ctx,
                                     const LegionCopy& copy,
                                     const SelectCopySrcInput& input,
                                     SelectCopySrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::speculate(const MapperContext ctx,
                           const LegionCopy& copy,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const LegionCopy& copy,
                                  const CopyProfilingInfo& input)
{
  // No profiling for copies yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const LegionCopy& copy,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  // TODO: Copies can have key stores in the future
  output.chosen_functor = find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::select_close_sources(const MapperContext ctx,
                                      const Close& close,
                                      const SelectCloseSrcInput& input,
                                      SelectCloseSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const Close& close,
                                  const CloseProfilingInfo& input)
{
  // No profiling yet for legate
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Close& close,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_acquire(const MapperContext ctx,
                             const Acquire& acquire,
                             const MapAcquireInput& input,
                             MapAcquireOutput& output)
{
  // Nothing to do
}

void BaseMapper::speculate(const MapperContext ctx,
                           const Acquire& acquire,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const Acquire& acquire,
                                  const AcquireProfilingInfo& input)
{
  // No profiling for legate yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Acquire& acquire,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_release(const MapperContext ctx,
                             const Release& release,
                             const MapReleaseInput& input,
                             MapReleaseOutput& output)
{
  // Nothing to do
}

void BaseMapper::select_release_sources(const MapperContext ctx,
                                        const Release& release,
                                        const SelectReleaseSrcInput& input,
                                        SelectReleaseSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::speculate(const MapperContext ctx,
                           const Release& release,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const Release& release,
                                  const ReleaseProfilingInfo& input)
{
  // No profiling for legate yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Release& release,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::select_partition_projection(const MapperContext ctx,
                                             const Partition& partition,
                                             const SelectPartitionProjectionInput& input,
                                             SelectPartitionProjectionOutput& output)
{
  // If we have an open complete partition then use it
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions[0];
  else
    output.chosen_partition = LogicalPartition::NO_PART;
}

void BaseMapper::map_partition(const MapperContext ctx,
                               const Partition& partition,
                               const MapPartitionInput& input,
                               MapPartitionOutput& output)
{
  Processor target_proc{Processor::NO_PROC};
  if (!local_omps.empty())
    target_proc = local_omps.front();
  else
    target_proc = local_cpus.front();

  auto store_target = default_store_targets(target_proc.kind()).front();

#ifdef DEBUG_LEGATE
  assert(partition.requirement.instance_fields.size() == 1);
#endif

  Store store(legion_runtime->get_mapper_runtime(), ctx, &partition.requirement);
  StoreMapping mapping = StoreMapping::default_mapping(store, store_target, false);

  auto reqs = mapping.requirements();
  output.chosen_instances.resize(1);
  PhysicalInstance& result = output.chosen_instances.front();
  bool can_fail            = false;
  while (map_legate_store(ctx, partition, mapping, reqs, target_proc, result, can_fail)) {
    if (result == PhysicalInstance()) break;
    if (runtime->acquire_instance(ctx, result)) {
#ifdef DEBUG_LEGATE
      logger.debug() << "Partition Op " << partition.get_unique_id() << ": acquired instance "
                     << result << " for reqs: 0";
#endif
      break;
    }
#ifdef DEBUG_LEGATE
    logger.debug() << "Partition Op " << partition.get_unique_id()
                   << ": failed to acquire instance " << result << " for reqs: 0";
#endif
    AutoLock lock(ctx, local_instances->manager_lock());
    local_instances->erase(result);
  }
}

void BaseMapper::select_partition_sources(const MapperContext ctx,
                                          const Partition& partition,
                                          const SelectPartitionSrcInput& input,
                                          SelectPartitionSrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const Partition& partition,
                                  const PartitionProfilingInfo& input)
{
  // No profiling yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Partition& partition,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Fill& fill,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = fill.is_index_space
                            ? find_sharding_functor_by_key_store_projection({fill.requirement})
                            : find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::configure_context(const MapperContext ctx,
                                   const LegionTask& task,
                                   ContextConfigOutput& output)
{
  // Use the defaults currently
}

void BaseMapper::select_tunable_value(const MapperContext ctx,
                                      const LegionTask& task,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  auto value   = tunable_value(input.tunable_id);
  output.size  = value.size();
  output.value = malloc(output.size);
  memcpy(output.value, value.ptr(), output.size);
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const MustEpoch& epoch,
                                         const SelectShardingFunctorInput& input,
                                         MustEpochShardingFunctorOutput& output)
{
  // No must epoch launches in legate
  LEGATE_ABORT;
}

void BaseMapper::memoize_operation(const MapperContext ctx,
                                   const Mappable& mappable,
                                   const MemoizeInput& input,
                                   MemoizeOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_must_epoch(const MapperContext ctx,
                                const MapMustEpochInput& input,
                                MapMustEpochOutput& output)
{
  // No must epoch launches in legate
  LEGATE_ABORT;
}

void BaseMapper::map_dataflow_graph(const MapperContext ctx,
                                    const MapDataflowGraphInput& input,
                                    MapDataflowGraphOutput& output)
{
  // Not supported yet
  LEGATE_ABORT;
}

void BaseMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  // Just map all the ready tasks
  for (auto task : input.ready_tasks) output.map_tasks.insert(task);
}

void BaseMapper::select_steal_targets(const MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output)
{
  // Nothing to do, no stealing in the leagte mapper currently
}

void BaseMapper::permit_steal_request(const MapperContext ctx,
                                      const StealRequestInput& input,
                                      StealRequestOutput& output)
{
  // Nothing to do, no stealing in the legate mapper currently
  LEGATE_ABORT;
}

void BaseMapper::handle_message(const MapperContext ctx, const MapperMessage& message)
{
  // We shouldn't be receiving any messages currently
  LEGATE_ABORT;
}

void BaseMapper::handle_task_result(const MapperContext ctx, const MapperTaskResult& result)
{
  // Nothing to do since we should never get one of these
  LEGATE_ABORT;
}

}  // namespace mapping
}  // namespace legate
