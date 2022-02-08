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

#include "legion/legion_mapping.h"

#include "core/data/store.h"
#include "core/mapping/base_mapper.h"
#include "core/mapping/instance_manager.h"
#include "core/mapping/task.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/utilities/linearize.h"
#include "legate_defines.h"

using LegionTask = Legion::Task;

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {
namespace mapping {

BaseMapper::BaseMapper(Runtime* rt, Machine m, const LibraryContext& ctx)
  : Mapper(rt->get_mapper_runtime()),
    legion_runtime(rt),
    machine(m),
    context(ctx),
    local_node(get_local_node()),
    total_nodes(get_total_nodes(m)),
    mapper_name(std::move(create_name(local_node))),
    logger(create_logger_name().c_str()),
    local_instances(std::make_unique<InstanceManager>())
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
      case Processor::IO_PROC: {
        local_ios.push_back(local_proc);
        break;
      }
      case Processor::PY_PROC: {
        local_pys.push_back(local_proc);
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
  if (show_usage != NULL) {
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

  // We never want valid instances
  switch (target) {
    case TaskTarget::CPU: {
      output.initial_proc = local_cpus.front();
      break;
    }
    case TaskTarget::GPU: {
      output.initial_proc = local_gpus.front();
      break;
    }
    case TaskTarget::OMP: {
      output.initial_proc = local_omps.front();
      break;
    }
  }
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

  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      round_robin(local_cpus);
      break;
    }
    case Processor::TOC_PROC: {
      round_robin(local_gpus);
      break;
    }
    case Processor::OMP_PROC: {
      round_robin(local_omps);
      break;
    }
    default: LEGATE_ABORT;
  }
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

  int32_t num_procs = 1;
  switch (kind) {
    case Processor::LOC_PROC: {
      num_procs = static_cast<int32_t>(local_cpus.size());
      break;
    }
    case Processor::TOC_PROC: {
      num_procs = static_cast<int32_t>(local_gpus.size());
      break;
    }
    case Processor::OMP_PROC: {
      num_procs = static_cast<int32_t>(local_omps.size());
      break;
    }
    default: LEGATE_ABORT;
  }

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

  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      distribute(local_cpus);
      break;
    }
    case Processor::TOC_PROC: {
      distribute(local_gpus);
      break;
    }
    case Processor::OMP_PROC: {
      distribute(local_omps);
      break;
    }
    default: LEGATE_ABORT;
  }
}

void BaseMapper::slice_task(const MapperContext ctx,
                            const LegionTask& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  if (task.tag == LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG)
    slice_manual_task(ctx, task, input, output);
  else
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
  // Should never be mapping the top-level task here
  assert(task.get_depth() > 0);

  // Let's populate easy outputs first
  output.chosen_variant = find_variant(ctx, task, task.target_proc.kind());
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);

  Task legate_task(&task, context, runtime, ctx);

  std::vector<StoreTarget> options;
  switch (task.target_proc.kind()) {
    case Processor::LOC_PROC: {
      options = {StoreTarget::SYSMEM};
      break;
    }
    case Processor::TOC_PROC: {
      options = {StoreTarget::FBMEM, StoreTarget::ZCMEM};
      break;
    }
    case Processor::OMP_PROC: {
      options = {StoreTarget::SOCKETMEM, StoreTarget::SYSMEM};
      break;
    }
    default: LEGATE_ABORT;
  }

  auto mappings = store_mappings(legate_task, options);

  std::map<RegionField::Id, uint32_t> client_mapped;
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
      if (store.is_future()) continue;

      auto& rf = store.region_field();
      auto key = rf.unique_id();

      auto finder = client_mapped.find(key);
      // If this is the first store mapping for this requirement,
      // we record the mapping index for future reference.
      if (finder == client_mapped.end()) client_mapped[key] = mapping_idx;
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
      if (store.is_future()) continue;
      auto key = store.region_field().unique_id();
      if (client_mapped.find(key) != client_mapped.end()) continue;
      client_mapped[key] = static_cast<int32_t>(mappings.size());
      mappings.push_back(StoreMapping::default_mapping(store, default_option, exact));
    }
  };

  generate_default_mappings(legate_task.inputs(), false);
  generate_default_mappings(legate_task.outputs(), false);
  generate_default_mappings(legate_task.reductions(), false);

  output.chosen_instances.resize(task.regions.size());

  // Map each field separately for each of the logical regions
  std::vector<PhysicalInstance> needed_acquires;
  std::map<PhysicalInstance, std::set<uint32_t>> instances_to_mappings;
  for (uint32_t mapping_idx = 0; mapping_idx < mappings.size(); ++mapping_idx) {
    auto& mapping    = mappings[mapping_idx];
    auto req_indices = mapping.requirement_indices();

    if (req_indices.empty()) continue;

    if (mapping.for_unbound_stores()) {
      for (auto req_idx : req_indices)
        output.output_targets[req_idx] = get_target_memory(task.target_proc, mapping.policy.target);
      continue;
    }

    std::vector<std::reference_wrapper<const RegionRequirement>> reqs;
    for (auto req_idx : req_indices) {
      const auto& req = task.regions[req_idx];
      if (!req.region.exists()) continue;
      reqs.push_back(std::cref(req));
    }

    if (reqs.empty()) continue;

    // Get the reference to our valid instances in case we decide to use them
    PhysicalInstance result;
    if (map_legate_store(ctx, task, mapping, reqs, task.target_proc, result))
      needed_acquires.push_back(result);

    for (auto req_idx : req_indices) output.chosen_instances[req_idx].push_back(result);
    instances_to_mappings[result].insert(mapping_idx);
  }

  // Do an acquire on all the instances so we have our result
  // Keep doing this until we succed or we get an out of memory error
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);

    for (auto failed_acquire : failed_acquires) {
      auto affected_mappings = instances_to_mappings[failed_acquire];
      instances_to_mappings.erase(failed_acquire);

      for (auto& mapping_idx : affected_mappings) {
        auto& mapping    = mappings[mapping_idx];
        auto req_indices = mapping.requirement_indices();

        std::vector<std::reference_wrapper<const RegionRequirement>> reqs;
        for (auto req_idx : req_indices) reqs.push_back(std::cref(task.regions[req_idx]));

        for (auto req_idx : req_indices) {
          auto& instances   = output.chosen_instances[req_idx];
          uint32_t inst_idx = 0;
          for (; inst_idx < instances.size(); ++inst_idx)
            if (instances[inst_idx] == failed_acquire) break;
          instances.erase(instances.begin() + inst_idx);
        }

        PhysicalInstance result;
        if (map_legate_store(ctx, task, mapping, reqs, task.target_proc, result))
          needed_acquires.push_back(result);

        for (auto req_idx : req_indices) output.chosen_instances[req_idx].push_back(result);
        instances_to_mappings[result].insert(mapping_idx);
      }
    }
  }
}

void BaseMapper::map_replicate_task(const MapperContext ctx,
                                    const LegionTask& task,
                                    const MapTaskInput& input,
                                    const MapTaskOutput& def_output,
                                    MapReplicateTaskOutput& output)
{
  LEGATE_ABORT;
}

bool BaseMapper::find_existing_instance(LogicalRegion region,
                                        FieldID fid,
                                        Memory target_memory,
                                        PhysicalInstance& result,
                                        Strictness strictness)
{
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
                                  std::vector<std::reference_wrapper<const RegionRequirement>> reqs,
                                  Processor target_proc,
                                  PhysicalInstance& result)
{
  const auto& policy = mapping.policy;
  std::vector<LogicalRegion> regions;
  for (auto& req : reqs) regions.push_back(req.get().region);
  auto target_memory = get_target_memory(target_proc, policy.target);

  ReductionOpID redop = 0;
  bool first          = true;
  for (auto& req : reqs) {
    if (first)
      redop = req.get().redop;
    else {
      if (redop != req.get().redop) {
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

    if (!runtime->create_physical_instance(
          ctx, target_memory, layout_constraints, regions, result, true /*acquire*/))
      report_failed_mapping(mappable, mapping.requirement_index(), target_memory, redop);
    // We already did the acquire
    return false;
  }

  auto& fields = layout_constraints.field_constraint.field_set;

  // See if we already have it in our local instances
  if (fields.size() == 1 && regions.size() == 1 &&
      local_instances->find_instance(
        regions.front(), fields.front(), target_memory, result, policy))
    // Needs acquire to keep the runtime happy
    return true;

  // This whole process has to appear atomic
  runtime->disable_reentrant(ctx);

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
    regions = group->regions;
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
    if (created)
      logger.info("%s created instance %lx containing %zd bytes in memory " IDFMT,
                  get_mapper_name(),
                  result.get_instance_id(),
                  footprint,
                  target_memory.id);
    // Only save the result for future use if it is not an external instance
    if (!result.is_external_instance() && group != nullptr) {
      assert(fields.size() == 1);
      auto fid = fields.front();
      local_instances->record_instance(group, fid, result, policy);
    }
    // We made it so no need for an acquire
    runtime->enable_reentrant(ctx);
    return false;
  }
  // Done with the atomic part
  runtime->enable_reentrant(ctx);

  // If we make it here then we failed entirely
  auto req_indices = mapping.requirement_indices();
  for (auto req_idx : req_indices) report_failed_mapping(mappable, req_idx, target_memory, redop);
  return true;
}

bool BaseMapper::map_raw_array(const MapperContext ctx,
                               const Mappable& mappable,
                               uint32_t index,
                               LogicalRegion region,
                               FieldID fid,
                               Memory target_memory,
                               Processor target_proc,
                               const std::vector<PhysicalInstance>& valid,
                               PhysicalInstance& result,
                               bool memoize_result,
                               ReductionOpID redop /*=0*/)
{
  // If we're making a reduction instance, we should just make it now
  if (redop != 0) {
    // Switch the target memory if we're going to a GPU because
    // Realm's DMA system still does not support reductions
    const std::vector<LogicalRegion> regions(1, region);
    LayoutConstraintSet layout_constraints;
    // No specialization
    layout_constraints.add_constraint(SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, redop));
    // SOA-C dimension ordering
    std::vector<DimensionKind> dimension_ordering(4);
    dimension_ordering[0] = DIM_Z;
    dimension_ordering[1] = DIM_Y;
    dimension_ordering[2] = DIM_X;
    dimension_ordering[3] = DIM_F;
    layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, false /*contiguous*/));
    // Constraint for the kind of memory
    layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
    // Make sure we have our field
    const std::vector<FieldID> fields(1, fid);
    layout_constraints.add_constraint(FieldConstraint(fields, true /*contiguous*/));
    if (!runtime->create_physical_instance(
          ctx, target_memory, layout_constraints, regions, result, true /*acquire*/))
      report_failed_mapping(mappable, index, target_memory, redop);
    // We already did the acquire
    return false;
  }
  // See if we already have it in our local instances
  if (local_instances->find_instance(region, fid, target_memory, result))
    // Needs acquire to keep the runtime happy
    return true;

  // There's a little asymmetry here between CPUs and GPUs for NUMA effects
  // For CPUs NUMA-effects are within a factor of 2X additional latency and
  // reduced bandwidth, so it's better to just use data where it is rather
  // than move it. For GPUs though, the difference between local framebuffer
  // and remote can be on the order of 800 GB/s versus 20 GB/s over NVLink
  // so it's better to move things local, so we'll always try to make a local
  // instance before checking for a nearby instance in a different GPU.
  if (target_proc.exists() && ((target_proc.kind() == Processor::LOC_PROC) ||
                               (target_proc.kind() == Processor::OMP_PROC))) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (auto memory : affinity_mems) {
      if (local_instances->find_instance(region, fid, memory, result))
        // Needs acquire to keep the runtime happy
        return true;
    }
  }
  // This whole process has to appear atomic
  runtime->disable_reentrant(ctx);
  // Haven't made this instance before, so make it now
  // We can do an interesting optimization here to try to reduce unnecessary
  // inter-memory copies. For logical regions that are overlapping we try
  // to accumulate as many as possible into one physical instance and use
  // that instance for all the tasks for the different regions.
  // First we have to see if there is anything we overlap with
  const IndexSpace is = region.get_index_space();
  const Domain domain = runtime->get_index_space_domain(ctx, is);
  auto group          = local_instances->find_region_group(region, domain, fid, target_memory);

  // We're going to need some of this constraint information no matter
  // which path we end up taking below
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA-C dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_Z;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_X;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, false /*contiguous*/));
  // Constraint for the kind of memory
  layout_constraints.add_constraint(MemoryConstraint(target_memory.kind()));
  // Make sure we have our field
  const std::vector<FieldID> fields(1, fid);
  layout_constraints.add_constraint(FieldConstraint(fields, true /*contiguous*/));

  bool created;
  size_t footprint;
  if (runtime->find_or_create_physical_instance(ctx,
                                                target_memory,
                                                layout_constraints,
                                                group->regions,
                                                result,
                                                created,
                                                true /*acquire*/,
                                                memoize_result ? GC_NEVER_PRIORITY : 0,
                                                false /*tight bounds*/,
                                                &footprint)) {
    // We succeeded in making the instance where we want it
    assert(result.exists());
    if (created)
      logger.info("%s created instance %lx containing %zd bytes in memory " IDFMT,
                  get_mapper_name(),
                  result.get_instance_id(),
                  footprint,
                  target_memory.id);
    // Only save the result for future use if it is not an external instance
    if (memoize_result && !result.is_external_instance()) {
      auto replaced = local_instances->record_instance(group, fid, result);
      for (auto& instance : replaced) {
        if (!instance.is_external_instance())
          runtime->set_garbage_collection_priority(ctx, instance, 0);
      }
    }
    // We made it so no need for an acquire
    runtime->enable_reentrant(ctx);
    return false;
  }
  // Done with the atomic part
  runtime->enable_reentrant(ctx);

  // If we get here it's because we failed to make the instance, we still
  // have a few more tricks that we can try
  // First see if we can find an existing valid instance that we can use
  // with affinity to our target processor
  if (!valid.empty())
    for (auto& instance : valid) {
      // If it doesn't have the field then we don't care
      if (instance.has_field(fid)) continue;
      if (!target_proc.exists() || machine.has_affinity(target_proc, instance.get_location())) {
        result = instance;
        return true;
      }
    }

  // Still couldn't find an instance, see if we can find any instances
  // in memories that are local to our node that we can use
  if (target_proc.exists()) {
    Machine::MemoryQuery affinity_mems(machine);
    affinity_mems.has_affinity_to(target_proc);
    for (auto mem : affinity_mems)
      if (local_instances->find_instance(region, fid, mem, result))
        // Needs acquire to keep the runtime happy
        return true;
  } else if (find_existing_instance(region, fid, target_memory, result)) {
    return true;
  }
  // If we make it here then we failed entirely
  report_failed_mapping(mappable, index, target_memory, redop);
  return true;
}

void BaseMapper::filter_failed_acquires(std::vector<PhysicalInstance>& needed_acquires,
                                        std::set<PhysicalInstance>& failed_acquires)
{
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
  switch (mappable.get_mappable_type()) {
    case Mappable::TASK_MAPPABLE: {
      const auto task = mappable.as_task();
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of task %s (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          task->get_task_name(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "task %s (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          task->get_task_name(),
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::COPY_MAPPABLE: {
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of copy (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "copy (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      break;
    }
    case Mappable::INLINE_MAPPABLE: {
      if (redop > 0)
        logger.error(
          "Mapper %s failed to map reduction (%d) region "
          "requirement %d of inline mapping (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          redop,
          index,
          mappable.get_unique_id(),
          memory_kinds[target_memory.kind()],
          target_memory.id);
      else
        logger.error(
          "Mapper %s failed to map region requirement %d of "
          "inline mapping (UID %lld) into %s memory " IDFMT,
          get_mapper_name(),
          index,
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

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const LegionTask& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  for (auto& req : task.regions)
    if (req.tag == LEGATE_CORE_KEY_STORE_TAG) {
      output.chosen_functor = find_sharding_functor_by_projection_functor(req.projection);
      return;
    }

  output.chosen_functor = 0;
}

void BaseMapper::map_inline(const MapperContext ctx,
                            const InlineMapping& inline_op,
                            const MapInlineInput& input,
                            MapInlineOutput& output)
{
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  const RegionRequirement& req               = inline_op.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  uint32_t index = 0;
  std::vector<PhysicalInstance> needed_acquires;
  for (auto fid : req.privilege_fields) {
    if (map_raw_array(ctx,
                      inline_op,
                      0,
                      req.region,
                      fid,
                      local_system_memory,
                      inline_op.parent_task->current_proc,
                      valid,
                      output.chosen_instances[index],
                      false /*memoize*/,
                      req.redop))
      needed_acquires.push_back(output.chosen_instances[index]);
    ++index;
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
    for (uint32_t idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) == failed_instances.end()) continue;
      // Now try to remap it
      if (map_raw_array(ctx,
                        inline_op,
                        0 /*idx*/,
                        req.region,
                        *fit,
                        local_system_memory,
                        inline_op.parent_task->current_proc,
                        valid,
                        output.chosen_instances[idx],
                        false /*memoize*/))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
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
                          const Copy& copy,
                          const MapCopyInput& input,
                          MapCopyOutput& output)
{
  // We should always be able to materialize instances of the things
  // we are copying so make concrete source instances
  std::vector<PhysicalInstance> needed_acquires;
  Memory target_memory = local_system_memory;
  /*
  if (copy.is_index_space) {
    // If we've got GPUs, assume we're using them
    if (!local_gpus.empty() || !local_omps.empty()) {
      const ShardingID sid          = select_sharding_functor(copy);
      NumPyShardingFunctor* functor = find_sharding_functor(sid);
      Domain sharding_domain        = copy.index_domain;
      if (copy.sharding_space.exists())
        sharding_domain = runtime->get_index_space_domain(ctx, copy.sharding_space);
      const uint32_t local_index =
        functor->localize(copy.index_point, sharding_domain, total_nodes, local_node);
      if (!local_gpus.empty()) {
        const Processor proc = local_gpus[local_index % local_gpus.size()];
        target_memory        = local_frame_buffers[proc];
      } else {
        const Processor proc = local_omps[local_index % local_omps.size()];
        target_memory        = local_numa_domains[proc];
      }
    }
  } else {
  */
  {
    // If we have just one local GPU then let's use it, otherwise punt to CPU
    // since it's not clear which one we should use
    if (local_frame_buffers.size() == 1) target_memory = local_frame_buffers.begin()->second;
  }

  auto map_stores = [&](auto idx, auto& req, auto& inputs, auto& outputs) {
    auto& region = req.region;
    outputs.resize(req.privilege_fields.size());
    const auto& valid  = inputs;
    uint32_t fidx      = 0;
    const bool memoize = req.privilege != LEGION_REDUCE;
    for (auto fid : req.privilege_fields) {
      if (req.redop != 0) {
        ++fidx;
        continue;
      }
      if (find_existing_instance(region, fid, target_memory, outputs[fidx]) ||
          map_raw_array(ctx,
                        copy,
                        idx,
                        region,
                        fid,
                        target_memory,
                        Processor::NO_PROC,
                        valid,
                        outputs[fidx],
                        memoize))
        needed_acquires.push_back(outputs[fidx]);
      ++fidx;
    }
  };

  auto dst_offset          = copy.src_requirements.size();
  auto src_indirect_offset = dst_offset + copy.dst_requirements.size();
  auto dst_indirect_offset = src_indirect_offset + copy.src_indirect_requirements.size();

  for (uint32_t idx = 0; idx < copy.src_requirements.size(); idx++) {
    map_stores(
      idx, copy.src_requirements[idx], input.src_instances[idx], output.src_instances[idx]);

    map_stores(idx + dst_offset,
               copy.dst_requirements[idx],
               input.dst_instances[idx],
               output.dst_instances[idx]);

    if (idx < copy.src_indirect_requirements.size()) {
      std::vector<PhysicalInstance> outputs;
      map_stores(idx + src_indirect_offset,
                 copy.src_indirect_requirements[idx],
                 input.src_indirect_instances[idx],
                 outputs);
      output.src_indirect_instances[idx] = outputs[0];
    }

    if (idx < copy.dst_indirect_requirements.size()) {
      std::vector<PhysicalInstance> outputs;
      map_stores(idx + dst_indirect_offset,
                 copy.dst_indirect_requirements[idx],
                 input.dst_indirect_instances[idx],
                 outputs);
      output.dst_indirect_instances[idx] = outputs[0];
    }
  }

  auto remap_stores = [&](auto idx, auto& req, auto& inputs, auto& outputs, auto& failed_acquires) {
    auto& region       = req.region;
    const auto& valid  = inputs;
    uint32_t fidx      = 0;
    const bool memoize = req.privilege != LEGION_REDUCE;
    for (auto fid : req.privilege_fields) {
      if (failed_acquires.find(outputs[fidx]) == failed_acquires.end()) {
        ++fidx;
        continue;
      }
      if (map_raw_array(ctx,
                        copy,
                        idx,
                        region,
                        fid,
                        target_memory,
                        Processor::NO_PROC,
                        valid,
                        outputs[fidx],
                        memoize))
        needed_acquires.push_back(outputs[fidx]);
      ++fidx;
    }
  };

  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    // If we failed to acquire any of the instances we need to prune them
    // out of the mapper's data structure so do that first
    std::set<PhysicalInstance> failed_acquires;
    filter_failed_acquires(needed_acquires, failed_acquires);

    // Now go through and try to remap region requirements with failed acquisitions
    for (uint32_t idx = 0; idx < copy.src_requirements.size(); idx++) {
      remap_stores(idx,
                   copy.src_requirements[idx],
                   input.src_instances[idx],
                   output.src_instances[idx],
                   failed_acquires);

      remap_stores(idx + dst_offset,
                   copy.dst_requirements[idx],
                   input.dst_instances[idx],
                   output.dst_instances[idx],
                   failed_acquires);
      if (idx < copy.src_indirect_requirements.size()) {
        std::vector<PhysicalInstance> outputs(1, output.src_indirect_instances[idx]);
        remap_stores(idx + src_indirect_offset,
                     copy.src_indirect_requirements[idx],
                     input.src_indirect_instances[idx],
                     outputs,
                     failed_acquires);
      }
      if (idx < copy.dst_indirect_requirements.size()) {
        std::vector<PhysicalInstance> outputs(1, output.dst_indirect_instances[idx]);
        remap_stores(idx + dst_indirect_offset,
                     copy.dst_indirect_requirements[idx],
                     input.dst_indirect_instances[idx],
                     outputs,
                     failed_acquires);
      }
    }
  }
}

void BaseMapper::select_copy_sources(const MapperContext ctx,
                                     const Copy& copy,
                                     const SelectCopySrcInput& input,
                                     SelectCopySrcOutput& output)
{
  legate_select_sources(ctx, input.target, input.source_instances, output.chosen_ranking);
}

void BaseMapper::speculate(const MapperContext ctx, const Copy& copy, SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const MapperContext ctx,
                                  const Copy& copy,
                                  const CopyProfilingInfo& input)
{
  // No profiling for copies yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Copy& copy,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = 0;
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
  const RegionRequirement& req = partition.requirement;
  output.chosen_instances.resize(req.privilege_fields.size());
  const std::vector<PhysicalInstance>& valid = input.valid_instances;
  std::vector<PhysicalInstance> needed_acquires;
  uint32_t fidx      = 0;
  const bool memoize = true;
  for (auto fid : req.privilege_fields) {
    if (find_existing_instance(req.region,
                               fid,
                               local_system_memory,
                               output.chosen_instances[fidx],
                               Strictness::strict) ||
        map_raw_array(ctx,
                      partition,
                      0,
                      req.region,
                      fid,
                      local_system_memory,
                      Processor::NO_PROC,
                      valid,
                      output.chosen_instances[fidx],
                      memoize)) {
      needed_acquires.push_back(output.chosen_instances[fidx]);
    }
    ++fidx;
  }
  while (!needed_acquires.empty() &&
         !runtime->acquire_and_filter_instances(ctx, needed_acquires, true /*filter on acquire*/)) {
    assert(!needed_acquires.empty());
    std::set<PhysicalInstance> failed_instances;
    filter_failed_acquires(needed_acquires, failed_instances);
    // Now go through all the fields for the instances and try and remap
    auto fit = req.privilege_fields.begin();
    for (uint32_t idx = 0; idx < output.chosen_instances.size(); idx++, fit++) {
      if (failed_instances.find(output.chosen_instances[idx]) == failed_instances.end()) continue;
      // Now try to remap it
      if (map_raw_array(ctx,
                        partition,
                        0 /*idx*/,
                        req.region,
                        *fit,
                        local_system_memory,
                        Processor::NO_PROC,
                        valid,
                        output.chosen_instances[idx],
                        memoize))
        needed_acquires.push_back(output.chosen_instances[idx]);
    }
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
  output.chosen_functor = 0;
}

void BaseMapper::select_sharding_functor(const MapperContext ctx,
                                         const Fill& fill,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = 0;
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
