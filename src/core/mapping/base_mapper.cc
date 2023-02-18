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

std::string log_mappable(const Legion::Mappable& mappable, bool prefix_only = false)
{
  static const std::map<Legion::MappableType, std::string> prefixes = {
    {LEGION_TASK_MAPPABLE, "Task "},
    {LEGION_COPY_MAPPABLE, "Copy "},
    {LEGION_INLINE_MAPPABLE, "Inline mapping "},
    {LEGION_PARTITION_MAPPABLE, "Partition "},
  };
  auto finder = prefixes.find(mappable.get_mappable_type());
#ifdef DEBUG_LEGATE
  assert(finder != prefixes.end());
#endif
  if (prefix_only) return finder->second;

  std::stringstream ss;
  ss << finder->second << mappable.get_unique_id();
  return ss.str();
}

}  // namespace

BaseMapper::BaseMapper(std::unique_ptr<LegateMapper> legate_mapper,
                       Legion::Runtime* rt,
                       Legion::Machine m,
                       const LibraryContext& ctx)
  : Mapper(rt->get_mapper_runtime()),
    legate_mapper_(std::move(legate_mapper)),
    legion_runtime(rt),
    machine(m),
    context(ctx),
    local_node(get_local_node()),
    total_nodes_(get_total_nodes(m)),
    mapper_name(std::move(create_name(local_node))),
    logger(create_logger_name().c_str()),
    local_instances(InstanceManager::get_instance_manager()),
    reduction_instances(ReductionInstanceManager::get_instance_manager())
{
  // Query to find all our local processors
  Legion::Machine::ProcessorQuery local_procs(machine);
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
  Legion::Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Legion::Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (auto& local_gpu : local_gpus) {
    Legion::Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(local_gpu);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[local_gpu] = local_framebuffer.first();
  }
  for (auto& local_omp : local_omps) {
    Legion::Machine::MemoryQuery local_numa(machine);
    local_numa.local_address_space();
    local_numa.only_kind(Memory::SOCKET_MEM);
    local_numa.best_affinity_to(local_omp);
    if (local_numa.count() > 0)  // if we have NUMA memories then use them
      local_numa_domains[local_omp] = local_numa.first();
    else  // Otherwise we just use the local system memory
      local_numa_domains[local_omp] = local_system_memory;
  }
  generate_prime_factors();

  legate_mapper_->set_machine(this);
}

BaseMapper::~BaseMapper()
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

/*static*/ Legion::AddressSpace BaseMapper::get_local_node()
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

/*static*/ size_t BaseMapper::get_total_nodes(Legion::Machine m)
{
  Legion::Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<Legion::AddressSpace> spaces;
  for (auto proc : query) spaces.insert(proc.address_space());
  return spaces.size();
}

std::string BaseMapper::create_name(Legion::AddressSpace node) const
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

const char* BaseMapper::get_mapper_name() const { return mapper_name.c_str(); }

Legion::Mapping::Mapper::MapperSyncModel BaseMapper::get_mapper_sync_model() const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void BaseMapper::select_task_options(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     TaskOptions& output)
{
#ifdef LEGATE_USE_COLLECTIVE
  for (uint32_t idx = 0; idx < task.regions.size(); ++idx) {
    auto& req = task.regions[idx];
    if (req.privilege & LEGION_WRITE_PRIV) continue;
    // Look up the projection for the input region. There are cases where
    // Legate libraries register their own projection functors that are
    // not recorded by Legate Core. So, handle the case when these functors
    // are not present and allow for them to be missing.
    auto projection = find_legate_projection_functor(req.projection, true /* allow_mising */);
    if ((req.handle_type == LEGION_SINGULAR_PROJECTION) ||
        (projection != nullptr && projection->is_collective())) {
      output.check_collective_regions.insert(idx);
    }
  }
#endif

  std::vector<TaskTarget> options;
  if (!local_gpus.empty() && has_variant(ctx, task, Processor::TOC_PROC))
    options.push_back(TaskTarget::GPU);
  if (!local_omps.empty() && has_variant(ctx, task, Processor::OMP_PROC))
    options.push_back(TaskTarget::OMP);
  options.push_back(TaskTarget::CPU);

  Task legate_task(&task, context, runtime, ctx);
  auto target = legate_mapper_->task_target(legate_task, options);

  dispatch(target, [&output](auto& procs) { output.initial_proc = procs.front(); });
  // We never want valid instances
  output.valid_instances = false;
}

void BaseMapper::premap_task(const Legion::Mapping::MapperContext ctx,
                             const Legion::Task& task,
                             const PremapTaskInput& input,
                             PremapTaskOutput& output)
{
  // NO-op since we know that all our futures should be mapped in the system memory
}

void BaseMapper::slice_auto_task(const Legion::Mapping::MapperContext ctx,
                                 const Legion::Task& task,
                                 const SliceTaskInput& input,
                                 SliceTaskOutput& output)
{
  Legion::ProjectionID projection = 0;
  for (auto& req : task.regions)
    if (req.tag == LEGATE_CORE_KEY_STORE_TAG) {
      projection = req.projection;
      break;
    }
  auto key_functor = find_legate_projection_functor(projection);

  // For multi-node cases we should already have been sharded so we
  // should just have one or a few points here on this node, so iterate
  // them and round-robin them across the local processors here
  output.slices.reserve(input.domain.get_volume());

  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists())
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);

  auto round_robin = [&](auto& procs) {
    auto lo = key_functor->project_point(sharding_domain.lo(), sharding_domain);
    auto hi = key_functor->project_point(sharding_domain.hi(), sharding_domain);
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
      auto p   = key_functor->project_point(itr.p, sharding_domain);
      auto idx = linearize(lo, hi, p);
      output.slices.push_back(TaskSlice(
        Domain(itr.p, itr.p), procs[idx % procs.size()], false /*recurse*/, false /*stealable*/));
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

const std::vector<int32_t> BaseMapper::get_processor_grid(Processor::Kind kind, int32_t ndim)
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

void BaseMapper::slice_manual_task(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   const SliceTaskInput& input,
                                   SliceTaskOutput& output)
{
  output.slices.reserve(input.domain.get_volume());

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

void BaseMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  if (task.tag == LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG)
    slice_manual_task(ctx, task, input, output);
  else
    slice_auto_task(ctx, task, input, output);
}

bool BaseMapper::has_variant(const Legion::Mapping::MapperContext ctx,
                             const Legion::Task& task,
                             Processor::Kind kind)
{
  return find_variant(ctx, task, kind).has_value();
}

std::optional<Legion::VariantID> BaseMapper::find_variant(const Legion::Mapping::MapperContext ctx,
                                                          const Legion::Task& task,
                                                          Processor::Kind kind)
{
  const VariantCacheKey key(task.task_id, kind);
  auto finder = variants.find(key);
  if (finder != variants.end()) return finder->second;

  // Haven't seen it before so let's look it up to make sure it exists
  std::vector<Legion::VariantID> avail_variants;
  runtime->find_valid_variants(ctx, key.first, avail_variants, key.second);
  std::optional<Legion::VariantID> result;
  for (auto vid : avail_variants) {
#ifdef DEBUG_LEGATE
    assert(vid > 0);
#endif
    switch (vid) {
      case LEGATE_CPU_VARIANT:
      case LEGATE_OMP_VARIANT:
      case LEGATE_GPU_VARIANT: {
        result = vid;
        break;
      }
      default: LEGATE_ABORT;  // unhandled variant kind
    }
  }
  variants[key] = result;
  return result;
}

void BaseMapper::map_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output)
{
#ifdef DEBUG_LEGATE
  logger.debug() << "Entering map_task for "
                 << Legion::Mapping::Utilities::to_string(runtime, ctx, task);
#endif

  // Should never be mapping the top-level task here
  assert(task.get_depth() > 0);

  // Let's populate easy outputs first
  auto variant = find_variant(ctx, task, task.target_proc.kind());
#ifdef DEBUG_LEGATE
  assert(variant.has_value());
#endif
  output.chosen_variant = *variant;
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);

  Task legate_task(&task, context, runtime, ctx);

  const auto& options = default_store_targets(task.target_proc.kind());

  auto mappings = legate_mapper_->store_mappings(legate_task, options);

  auto validate_colocation = [this](const auto& mapping) {
    if (mapping.stores.empty()) {
      logger.error("Store mapping must contain at least one store");
      LEGATE_ABORT;
    }
    if (mapping.stores.size() > 1 && mapping.policy.ordering.relative) {
      logger.error("Colocation with relative dimension ordering is illegal");
      LEGATE_ABORT;
    }
    auto& first_store = mapping.stores.front();
    for (auto it = mapping.stores.begin() + 1; it != mapping.stores.end(); ++it) {
      if (!it->can_colocate_with(first_store)) {
        logger.error("Mapper %s tried to colocate stores that cannot colocate", get_mapper_name());
        LEGATE_ABORT;
      }
    }
    assert(!(mapping.for_future() || mapping.for_unbound_store()) || mapping.stores.size() == 1);
  };

#ifdef DEBUG_LEGATE
  for (auto& mapping : mappings) validate_colocation(mapping);
#endif

  std::vector<StoreMapping> for_futures, for_unbound_stores, for_stores;
  std::set<uint32_t> mapped_futures;
  std::set<RegionField::Id> mapped_regions;

  for (auto& mapping : mappings) {
    if (mapping.for_future()) {
      mapped_futures.insert(mapping.store().future_index());
      for_futures.push_back(std::move(mapping));
    } else if (mapping.for_unbound_store()) {
      mapped_regions.insert(mapping.store().unique_region_field_id());
      for_unbound_stores.push_back(std::move(mapping));
    } else {
      for (auto& store : mapping.stores) mapped_regions.insert(store.unique_region_field_id());
      for_stores.push_back(std::move(mapping));
    }
  }

  auto check_consistency = [this](const auto& mappings) {
    std::map<RegionField::Id, InstanceMappingPolicy> policies;
    for (const auto& mapping : mappings)
      for (auto& store : mapping.stores) {
        auto key    = store.unique_region_field_id();
        auto finder = policies.find(key);
        if (policies.end() == finder)
          policies[key] = mapping.policy;
        else if (mapping.policy != finder->second) {
          logger.error("Mapper %s returned inconsistent store mappings", get_mapper_name());
          LEGATE_ABORT;
        }
      }
  };
#ifdef DEBUG_LEGATE
  check_consistency(for_stores);
#endif

  // Generate default mappings for stores that are not yet mapped by the client mapper
  auto default_option            = options.front();
  auto generate_default_mappings = [&](auto& stores, bool exact) {
    for (auto& store : stores) {
      auto mapping = StoreMapping::default_mapping(store, default_option, exact);
      if (store.is_future()) {
        auto fut_idx = store.future_index();
        if (mapped_futures.find(fut_idx) != mapped_futures.end()) continue;
        mapped_futures.insert(fut_idx);
        for_futures.push_back(std::move(mapping));
      } else {
        auto key = store.unique_region_field_id();
        if (mapped_regions.find(key) != mapped_regions.end()) continue;
        mapped_regions.insert(key);
        if (store.unbound())
          for_unbound_stores.push_back(std::move(mapping));
        else
          for_stores.push_back(std::move(mapping));
      }
    }
  };
  generate_default_mappings(legate_task.inputs(), false);
  generate_default_mappings(legate_task.outputs(), false);
  generate_default_mappings(legate_task.reductions(), false);

  // Map future-backed stores
  auto map_futures = [&](auto& mappings) {
    for (auto& mapping : mappings) {
      StoreTarget target = mapping.policy.target;
#ifdef LEGATE_NO_FUTURES_ON_FB
      if (target == StoreTarget::FBMEM) target = StoreTarget::ZCMEM;
#endif
      output.future_locations.push_back(get_target_memory(task.target_proc, target));
    }
  };
  map_futures(for_futures);

  // Map unbound stores
  auto map_unbound_stores = [&](auto& mappings) {
    for (auto& mapping : mappings) {
      auto req_idx                   = mapping.requirement_index();
      output.output_targets[req_idx] = get_target_memory(task.target_proc, mapping.policy.target);
      auto ndim                      = mapping.store().dim();
      // FIXME: Unbound stores can have more than one dimension later
      std::vector<Legion::DimensionKind> dimension_ordering;
      for (int32_t dim = ndim - 1; dim >= 0; --dim)
        dimension_ordering.push_back(static_cast<Legion::DimensionKind>(
          static_cast<int32_t>(Legion::DimensionKind::LEGION_DIM_X) + dim));
      dimension_ordering.push_back(Legion::DimensionKind::LEGION_DIM_F);
      output.output_constraints[req_idx].ordering_constraint =
        Legion::OrderingConstraint(dimension_ordering, false);
    }
  };
  map_unbound_stores(for_unbound_stores);

  output.chosen_instances.resize(task.regions.size());
  std::map<const Legion::RegionRequirement*, std::vector<Legion::Mapping::PhysicalInstance>*>
    output_map;
  for (uint32_t idx = 0; idx < task.regions.size(); ++idx)
    output_map[&task.regions[idx]] = &output.chosen_instances[idx];

  map_legate_stores(ctx, task, for_stores, task.target_proc, output_map);
}

void BaseMapper::map_replicate_task(const Legion::Mapping::MapperContext ctx,
                                    const Legion::Task& task,
                                    const MapTaskInput& input,
                                    const MapTaskOutput& def_output,
                                    MapReplicateTaskOutput& output)
{
  LEGATE_ABORT;
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

void BaseMapper::map_legate_stores(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Mappable& mappable,
                                   std::vector<StoreMapping>& mappings,
                                   Processor target_proc,
                                   OutputMap& output_map)
{
  auto try_mapping = [&](bool can_fail) {
    const Legion::Mapping::PhysicalInstance NO_INST{};
    std::vector<Legion::Mapping::PhysicalInstance> instances;
    for (auto& mapping : mappings) {
      Legion::Mapping::PhysicalInstance result = NO_INST;
      auto reqs                                = mapping.requirements();
      while (map_legate_store(ctx, mappable, mapping, reqs, target_proc, result, can_fail)) {
        if (NO_INST == result) {
#ifdef DEBUG_LEGATE
          assert(can_fail);
#endif
          for (auto& instance : instances) runtime->release_instance(ctx, instance);
          return false;
        }
#ifdef DEBUG_LEGATE
        std::stringstream reqs_ss;
        for (auto req_idx : mapping.requirement_indices()) reqs_ss << " " << req_idx;
#endif
        if (runtime->acquire_instance(ctx, result)) {
#ifdef DEBUG_LEGATE
          logger.debug() << log_mappable(mappable) << ": acquired instance " << result
                         << " for reqs:" << reqs_ss.str();
#endif
          break;
        }
#ifdef DEBUG_LEGATE
        logger.debug() << log_mappable(mappable) << ": failed to acquire instance " << result
                       << " for reqs:" << reqs_ss.str();
#endif
        if ((*reqs.begin())->redop != 0) {
          Legion::Mapping::AutoLock lock(ctx, reduction_instances->manager_lock());
          reduction_instances->erase(result);
        } else {
          Legion::Mapping::AutoLock lock(ctx, local_instances->manager_lock());
          local_instances->erase(result);
        }
        result = NO_INST;
      }
      instances.push_back(result);
    }

    // If we're here, all stores are mapped and instances are all acquired
    for (uint32_t idx = 0; idx < mappings.size(); ++idx) {
      auto& mapping  = mappings[idx];
      auto& instance = instances[idx];
      for (auto& req : mapping.requirements()) output_map[req]->push_back(instance);
    }
    return true;
  };

  // We can retry the mapping with tightened policies only if at least one of the policies
  // is lenient
  bool can_fail = false;
  for (auto& mapping : mappings) can_fail = can_fail || !mapping.policy.exact;

  if (!try_mapping(can_fail)) {
#ifdef DEBUG_LEGATE
    logger.debug() << log_mappable(mappable) << " failed to map all stores, retrying with "
                   << "tighter policies";
#endif
    // If instance creation failed we try mapping all stores again, but request tight instances for
    // write requirements. The hope is that these write requirements cover the entire region (i.e.
    // they use a complete partition), so the new tight instances will invalidate any pre-existing
    // "bloated" instances for the same region, freeing up enough memory so that mapping can succeed
    tighten_write_policies(mappable, mappings);
    try_mapping(false);
  }
}

void BaseMapper::tighten_write_policies(const Legion::Mappable& mappable,
                                        std::vector<StoreMapping>& mappings)
{
  for (auto& mapping : mappings) {
    // If the policy is exact, there's nothing we can tighten
    if (mapping.policy.exact) continue;

    int32_t priv = LEGION_NO_ACCESS;
    for (auto* req : mapping.requirements()) priv |= req->privilege;
    // We tighten only write requirements
    if (!(priv & LEGION_WRITE_PRIV)) continue;

#ifdef DEBUG_LEGATE
    std::stringstream reqs_ss;
    for (auto req_idx : mapping.requirement_indices()) reqs_ss << " " << req_idx;
    logger.debug() << log_mappable(mappable)
                   << ": tightened mapping policy for reqs:" << reqs_ss.str();
#endif
    mapping.policy.exact = true;
  }
}

bool BaseMapper::map_legate_store(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Mappable& mappable,
                                  const StoreMapping& mapping,
                                  const std::set<const Legion::RegionRequirement*>& reqs,
                                  Processor target_proc,
                                  Legion::Mapping::PhysicalInstance& result,
                                  bool can_fail)
{
  if (reqs.empty()) return false;

  const auto& policy = mapping.policy;
  std::vector<Legion::LogicalRegion> regions;
  for (auto* req : reqs) regions.push_back(req->region);
  auto target_memory = get_target_memory(target_proc, policy.target);

  auto redop = (*reqs.begin())->redop;
#ifdef DEBUG_LEGATE
  for (auto* req : reqs) {
    if (redop != req->redop) {
      logger.error(
        "Colocated stores should be either non-reduction arguments "
        "or reductions with the same reduction operator.");
      LEGATE_ABORT;
    }
  }
#endif

  // Generate layout constraints from the store mapping
  Legion::LayoutConstraintSet layout_constraints;
  mapping.populate_layout_constraints(layout_constraints);
  auto& fields = layout_constraints.field_constraint.field_set;

  // If we're making a reduction instance:
  if (redop != 0) {
    // We need to hold the instance manager lock as we're about to try
    // to find an instance
    Legion::Mapping::AutoLock reduction_lock(ctx, reduction_instances->manager_lock());

    // This whole process has to appear atomic
    runtime->disable_reentrant(ctx);

    // reuse reductions only for GPU tasks:
    if (target_proc.kind() == Processor::TOC_PROC) {
      // See if we already have it in our local instances
      if (fields.size() == 1 && regions.size() == 1 &&
          reduction_instances->find_instance(
            redop, regions.front(), fields.front(), target_memory, result, policy)) {
#ifdef DEBUG_LEGATE
        logger.debug() << "Operation " << mappable.get_unique_id()
                       << ": reused cached reduction instance " << result << " for "
                       << regions.front();
#endif
        runtime->enable_reentrant(ctx);
        // Needs acquire to keep the runtime happy
        return true;
      }
    }

    // if we didn't find it, create one
    layout_constraints.add_constraint(
      Legion::SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, redop));
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
      for (auto& r : regions) msg << " " << r;
      msg << " (size: " << footprint << " bytes, memory: " << target_memory << ")";
#endif
      if (target_proc.kind() == Processor::TOC_PROC) {
        // store reduction instance
        if (fields.size() == 1 && regions.size() == 1) {
          auto fid = fields.front();
          reduction_instances->record_instance(redop, regions.front(), fid, result, policy);
        }
      }
      runtime->enable_reentrant(ctx);
      // We already did the acquire
      return false;
    }
    if (!can_fail)
      report_failed_mapping(mappable, mapping.requirement_index(), target_memory, redop);
    return true;
  }

  Legion::Mapping::AutoLock lock(ctx, local_instances->manager_lock());
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
    auto is             = regions.front().get_index_space();
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

void BaseMapper::report_failed_mapping(const Legion::Mappable& mappable,
                                       uint32_t index,
                                       Memory target_memory,
                                       Legion::ReductionOpID redop)
{
  static const char* memory_kinds[] = {
#define MEM_NAMES(name, desc) desc,
    REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
  };

  std::string opname = "";
  if (mappable.get_mappable_type() == Legion::Mappable::TASK_MAPPABLE) {
    const auto task = mappable.as_task();
    opname          = task->get_task_name();
  }

  std::string provenance = mappable.get_provenance_string();
  if (provenance.empty()) provenance = "unknown provenance";

  std::stringstream req_ss;
  if (redop > 0)
    req_ss << "reduction (" << redop << ") requirement " << index;
  else
    req_ss << "region requirement " << index;

  logger.error("Mapper %s failed to map %s of %s%s[%s] (UID %lld) into %s memory " IDFMT,
               get_mapper_name(),
               req_ss.str().c_str(),
               log_mappable(mappable, true /*prefix_only*/).c_str(),
               opname.c_str(),
               provenance.c_str(),
               mappable.get_unique_id(),
               memory_kinds[target_memory.kind()],
               target_memory.id);
  LEGATE_ABORT;
}

void BaseMapper::select_task_variant(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     const SelectVariantInput& input,
                                     SelectVariantOutput& output)
{
  auto variant = find_variant(ctx, task, input.processor.kind());
#ifdef DEBUG_LEGATE
  assert(variant.has_value());
#endif
  output.chosen_variant = *variant;
}

void BaseMapper::postmap_task(const Legion::Mapping::MapperContext ctx,
                              const Legion::Task& task,
                              const PostMapInput& input,
                              PostMapOutput& output)
{
  // We should currently never get this call in Legate
  LEGATE_ABORT;
}

void BaseMapper::select_task_sources(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     const SelectTaskSrcInput& input,
                                     SelectTaskSrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void add_instance_to_band_ranking(
  const Legion::Mapping::PhysicalInstance& instance,
  const Legion::AddressSpace& local_node,
  std::map<Memory, uint32_t /*bandwidth*/>& source_memories,
  std::vector<std::pair<Legion::Mapping::PhysicalInstance, uint32_t>>& band_ranking,
  const Memory& destination_memory,
  const Legion::Machine& machine)
{
  Memory location = instance.get_location();
  auto finder     = source_memories.find(location);
  if (finder == source_memories.end()) {
    std::vector<Legion::MemoryMemoryAffinity> affinity;
    machine.get_mem_mem_affinity(
      affinity, location, destination_memory, false /*not just local affinities*/);
    uint32_t memory_bandwidth = 0;
    if (!affinity.empty()) {
#ifdef DEBUG_LEGATE
      assert(affinity.size() == 1);
#endif
      memory_bandwidth = affinity[0].bandwidth;
    }
    source_memories[location] = memory_bandwidth;
    band_ranking.push_back(
      std::pair<Legion::Mapping::PhysicalInstance, uint32_t>(instance, memory_bandwidth));
  } else
    band_ranking.push_back(
      std::pair<Legion::Mapping::PhysicalInstance, uint32_t>(instance, finder->second));
}

void BaseMapper::legate_select_sources(
  const Legion::Mapping::MapperContext ctx,
  const Legion::Mapping::PhysicalInstance& target,
  const std::vector<Legion::Mapping::PhysicalInstance>& sources,
  const std::vector<Legion::Mapping::CollectiveView>& collective_sources,
  std::deque<Legion::Mapping::PhysicalInstance>& ranking)
{
  std::map<Memory, uint32_t /*bandwidth*/> source_memories;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination.
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory destination_memory = target.get_location();
  // fill in a vector of the sources with their bandwidths and sort them
  std::vector<std::pair<Legion::Mapping::PhysicalInstance, uint32_t /*bandwidth*/>> band_ranking;
  for (uint32_t idx = 0; idx < sources.size(); idx++) {
    const Legion::Mapping::PhysicalInstance& instance = sources[idx];
    add_instance_to_band_ranking(
      instance, local_node, source_memories, band_ranking, destination_memory, machine);
  }

  for (uint32_t idx = 0; idx < collective_sources.size(); idx++) {
    std::vector<Legion::Mapping::PhysicalInstance> col_instances;
    collective_sources[idx].find_instances_nearest_memory(destination_memory, col_instances);
#ifdef DEBUG_LEGATE
    // there must exist at least one instance in the collective view
    assert(!col_instances.empty());
#endif
    // we need only first instance if there are several
    const Legion::Mapping::PhysicalInstance& instance = col_instances[0];
    add_instance_to_band_ranking(
      instance, local_node, source_memories, band_ranking, destination_memory, machine);
  }
#ifdef DEBUG_LEGATE
  assert(!band_ranking.empty());
#endif
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

void BaseMapper::speculate(const Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Task& task,
                                  const TaskProfilingInfo& input)
{
  // Shouldn't get any profiling feedback currently
  LEGATE_ABORT;
}

Legion::ShardingID BaseMapper::find_sharding_functor_by_key_store_projection(
  const std::vector<Legion::RegionRequirement>& requirements)
{
  Legion::ProjectionID proj_id = 0;
  for (auto& requirement : requirements)
    if (LEGATE_CORE_KEY_STORE_TAG == requirement.tag) {
      proj_id = requirement.projection;
      break;
    }
  return find_sharding_functor_by_projection_functor(proj_id);
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = task.is_index_space
                            ? find_sharding_functor_by_key_store_projection(task.regions)
                            : find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::map_inline(const Legion::Mapping::MapperContext ctx,
                            const Legion::InlineMapping& inline_op,
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
  std::vector<StoreMapping> mappings;
  mappings.push_back(StoreMapping::default_mapping(store, store_target, false));

  std::map<const Legion::RegionRequirement*, std::vector<Legion::Mapping::PhysicalInstance>*>
    output_map;
  for (auto* req : mappings.front().requirements()) output_map[req] = &output.chosen_instances;

  map_legate_stores(ctx, inline_op, mappings, target_proc, output_map);
}

void BaseMapper::select_inline_sources(const Legion::Mapping::MapperContext ctx,
                                       const Legion::InlineMapping& inline_op,
                                       const SelectInlineSrcInput& input,
                                       SelectInlineSrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::InlineMapping& inline_op,
                                  const InlineProfilingInfo& input)
{
  // No profiling yet for inline mappings
  LEGATE_ABORT;
}

void BaseMapper::map_copy(const Legion::Mapping::MapperContext ctx,
                          const Legion::Copy& copy,
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
    auto lo           = key_functor->project_point(sharding_domain.lo(), sharding_domain);
    auto hi           = key_functor->project_point(sharding_domain.hi(), sharding_domain);
    auto p            = key_functor->project_point(copy.index_point, sharding_domain);
    proc_id           = linearize(lo, hi, p);
  }
  if (!local_gpus.empty())
    target_proc = local_gpus[proc_id % local_gpus.size()];
  else if (!local_omps.empty())
    target_proc = local_omps[proc_id % local_omps.size()];
  else
    target_proc = local_cpus[proc_id % local_cpus.size()];

  auto store_target = default_store_targets(target_proc.kind()).front();

  // If we're mapping an indirect copy and have data resident in GPU memory,
  // map everything to CPU memory, as indirect copies on GPUs are currently
  // extremely slow.
  auto indirect =
    !copy.src_indirect_requirements.empty() || !copy.dst_indirect_requirements.empty();
  if (indirect && target_proc.kind() == Processor::TOC_PROC) {
    target_proc  = local_cpus.front();
    store_target = StoreTarget::SYSMEM;
  }

  Copy legate_copy(&copy, runtime, ctx);

  std::map<const Legion::RegionRequirement*, std::vector<Legion::Mapping::PhysicalInstance>*>
    output_map;
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

  map_legate_stores(ctx, copy, mappings, target_proc, output_map);
}

void BaseMapper::select_copy_sources(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Copy& copy,
                                     const SelectCopySrcInput& input,
                                     SelectCopySrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::speculate(const Legion::Mapping::MapperContext ctx,
                           const Legion::Copy& copy,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Copy& copy,
                                  const CopyProfilingInfo& input)
{
  // No profiling for copies yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Copy& copy,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  // TODO: Copies can have key stores in the future
  output.chosen_functor = find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::select_close_sources(const Legion::Mapping::MapperContext ctx,
                                      const Legion::Close& close,
                                      const SelectCloseSrcInput& input,
                                      SelectCloseSrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Close& close,
                                  const CloseProfilingInfo& input)
{
  // No profiling yet for legate
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Close& close,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_acquire(const Legion::Mapping::MapperContext ctx,
                             const Legion::Acquire& acquire,
                             const MapAcquireInput& input,
                             MapAcquireOutput& output)
{
  // Nothing to do
}

void BaseMapper::speculate(const Legion::Mapping::MapperContext ctx,
                           const Legion::Acquire& acquire,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Acquire& acquire,
                                  const AcquireProfilingInfo& input)
{
  // No profiling for legate yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Acquire& acquire,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_release(const Legion::Mapping::MapperContext ctx,
                             const Legion::Release& release,
                             const MapReleaseInput& input,
                             MapReleaseOutput& output)
{
  // Nothing to do
}

void BaseMapper::select_release_sources(const Legion::Mapping::MapperContext ctx,
                                        const Legion::Release& release,
                                        const SelectReleaseSrcInput& input,
                                        SelectReleaseSrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::speculate(const Legion::Mapping::MapperContext ctx,
                           const Legion::Release& release,
                           SpeculativeOutput& output)
{
  output.speculate = false;
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Release& release,
                                  const ReleaseProfilingInfo& input)
{
  // No profiling for legate yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Release& release,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::select_partition_projection(const Legion::Mapping::MapperContext ctx,
                                             const Legion::Partition& partition,
                                             const SelectPartitionProjectionInput& input,
                                             SelectPartitionProjectionOutput& output)
{
  // If we have an open complete partition then use it
  if (!input.open_complete_partitions.empty())
    output.chosen_partition = input.open_complete_partitions[0];
  else
    output.chosen_partition = Legion::LogicalPartition::NO_PART;
}

void BaseMapper::map_partition(const Legion::Mapping::MapperContext ctx,
                               const Legion::Partition& partition,
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
  std::vector<StoreMapping> mappings;
  mappings.push_back(StoreMapping::default_mapping(store, store_target, false));

  std::map<const Legion::RegionRequirement*, std::vector<Legion::Mapping::PhysicalInstance>*>
    output_map;
  for (auto* req : mappings.front().requirements()) output_map[req] = &output.chosen_instances;

  map_legate_stores(ctx, partition, mappings, target_proc, output_map);
}

void BaseMapper::select_partition_sources(const Legion::Mapping::MapperContext ctx,
                                          const Legion::Partition& partition,
                                          const SelectPartitionSrcInput& input,
                                          SelectPartitionSrcOutput& output)
{
  legate_select_sources(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Partition& partition,
                                  const PartitionProfilingInfo& input)
{
  // No profiling yet
  LEGATE_ABORT;
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Partition& partition,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Fill& fill,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = fill.is_index_space
                            ? find_sharding_functor_by_key_store_projection({fill.requirement})
                            : find_sharding_functor_by_projection_functor(0);
}

void BaseMapper::configure_context(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   ContextConfigOutput& output)
{
  // Use the defaults currently
}

void BaseMapper::select_tunable_value(const Legion::Mapping::MapperContext ctx,
                                      const Legion::Task& task,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  auto value   = legate_mapper_->tunable_value(input.tunable_id);
  output.size  = value.size();
  output.value = malloc(output.size);
  memcpy(output.value, value.ptr(), output.size);
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::MustEpoch& epoch,
                                         const SelectShardingFunctorInput& input,
                                         MustEpochShardingFunctorOutput& output)
{
  // No must epoch launches in legate
  LEGATE_ABORT;
}

void BaseMapper::memoize_operation(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Mappable& mappable,
                                   const MemoizeInput& input,
                                   MemoizeOutput& output)
{
  LEGATE_ABORT;
}

void BaseMapper::map_must_epoch(const Legion::Mapping::MapperContext ctx,
                                const MapMustEpochInput& input,
                                MapMustEpochOutput& output)
{
  // No must epoch launches in legate
  LEGATE_ABORT;
}

void BaseMapper::map_dataflow_graph(const Legion::Mapping::MapperContext ctx,
                                    const MapDataflowGraphInput& input,
                                    MapDataflowGraphOutput& output)
{
  // Not supported yet
  LEGATE_ABORT;
}

void BaseMapper::select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  // Just map all the ready tasks
  for (auto task : input.ready_tasks) output.map_tasks.insert(task);
}

void BaseMapper::select_steal_targets(const Legion::Mapping::MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output)
{
  // Nothing to do, no stealing in the leagte mapper currently
}

void BaseMapper::permit_steal_request(const Legion::Mapping::MapperContext ctx,
                                      const StealRequestInput& input,
                                      StealRequestOutput& output)
{
  // Nothing to do, no stealing in the legate mapper currently
  LEGATE_ABORT;
}

void BaseMapper::handle_message(const Legion::Mapping::MapperContext ctx,
                                const MapperMessage& message)
{
  // We shouldn't be receiving any messages currently
  LEGATE_ABORT;
}

void BaseMapper::handle_task_result(const Legion::Mapping::MapperContext ctx,
                                    const MapperTaskResult& result)
{
  // Nothing to do since we should never get one of these
  LEGATE_ABORT;
}

}  // namespace mapping
}  // namespace legate
