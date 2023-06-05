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
    {Processor::Kind::TOC_PROC, {StoreTarget::FBMEM, StoreTarget::ZCMEM}},
    {Processor::Kind::OMP_PROC, {StoreTarget::SOCKETMEM, StoreTarget::SYSMEM}},
    {Processor::Kind::LOC_PROC, {StoreTarget::SYSMEM}},
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

BaseMapper::BaseMapper(mapping::Mapper* legate_mapper,
                       Legion::Runtime* rt,
                       Legion::Machine m,
                       const LibraryContext* ctx)
  : Mapper(rt->get_mapper_runtime()),
    legate_mapper_(legate_mapper),
    legion_runtime(rt),
    legion_machine(m),
    context(ctx),
    logger(create_logger_name().c_str()),
    local_instances(InstanceManager::get_instance_manager()),
    reduction_instances(ReductionInstanceManager::get_instance_manager()),
    machine(m)
{
  std::stringstream ss;
  ss << context->get_library_name() << " on Node " << machine.local_node;
  mapper_name = ss.str();

  legate_mapper_->set_machine(this);
}

BaseMapper::~BaseMapper()
{
  // Compute the size of all our remaining instances in each memory
  const char* show_usage = getenv("LEGATE_SHOW_USAGE");
  if (show_usage != nullptr && atoi(show_usage) > 0) {
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
        context->get_library_name().c_str(),
        pair.second,
        memory_kinds[mem.kind()],
        mem.id,
        capacity,
        100.0 * double(pair.second) / capacity);
    }
  }
}

std::string BaseMapper::create_logger_name() const
{
  std::stringstream ss;
  ss << context->get_library_name() << ".mapper";
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

  Task legate_task(&task, context, runtime, ctx);
  auto& machine_desc = legate_task.machine_desc();
  auto all_targets   = machine_desc.valid_targets();

  std::vector<TaskTarget> options;
  for (auto& target : all_targets)
    if (has_variant(ctx, task, target)) options.push_back(target);
  if (options.empty()) {
    logger.error() << "Task " << task.get_task_name() << "[" << task.get_provenance_string()
                   << "] does not have a valid variant "
                   << "for this resource configuration: " << machine_desc;
    LEGATE_ABORT;
  }

  auto target = legate_mapper_->task_target(legate_task, options);
  // The initial processor just needs to have the same kind as the eventual target of this task
  output.initial_proc = machine.procs(target).front();

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

void BaseMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  Task legate_task(&task, context, runtime, ctx);

  auto& machine_desc = legate_task.machine_desc();
  auto local_range   = machine.slice(legate_task.target(), machine_desc);

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

  auto lo = key_functor->project_point(sharding_domain.lo(), sharding_domain);
  auto hi = key_functor->project_point(sharding_domain.hi(), sharding_domain);
  for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
    auto p   = key_functor->project_point(itr.p, sharding_domain);
    auto idx = linearize(lo, hi, p);
    output.slices.push_back(
      TaskSlice(Domain(itr.p, itr.p), local_range[idx], false /*recurse*/, false /*stealable*/));
  }
}

bool BaseMapper::has_variant(const Legion::Mapping::MapperContext ctx,
                             const Legion::Task& task,
                             TaskTarget target)
{
  return find_variant(ctx, task, to_kind(target)).has_value();
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

  Task legate_task(&task, context, runtime, ctx);

  if (task.is_index_space)
    // If this is an index task, point tasks already have the right targets, so we just need to
    // copy them to the mapper output
    output.target_procs.push_back(task.target_proc);
  else {
    // If this is a single task, here is the right place to compute the final target processor
    auto local_range =
      machine.slice(legate_task.target(), legate_task.machine_desc(), task.local_function);
#ifdef DEBUG_LEGATE
    assert(!local_range.empty());
#endif
    output.target_procs.push_back(local_range.first());
  }

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
      if (!it->get().can_colocate_with(first_store)) {
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
      auto fut_idx = mapping.store().future_index();
      // Only need to map Future-backed Stores corresponding to inputs (i.e. one of task.futures)
      if (fut_idx >= task.futures.size()) continue;
      if (mapped_futures.count(fut_idx) > 0) {
        logger.error("Mapper %s returned duplicate store mappings", get_mapper_name());
        LEGATE_ABORT;
      }
      mapped_futures.insert(fut_idx);
      for_futures.push_back(std::move(mapping));
    } else if (mapping.for_unbound_store()) {
      mapped_regions.insert(mapping.store().unique_region_field_id());
      for_unbound_stores.push_back(std::move(mapping));
    } else {
      for (auto& store : mapping.stores)
        mapped_regions.insert(store.get().unique_region_field_id());
      for_stores.push_back(std::move(mapping));
    }
  }

  auto check_consistency = [this](const auto& mappings) {
    std::map<RegionField::Id, InstanceMappingPolicy> policies;
    for (const auto& mapping : mappings)
      for (auto& store : mapping.stores) {
        auto key    = store.get().unique_region_field_id();
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
        // Only need to map Future-backed Stores corresponding to inputs (i.e. one of task.futures)
        if (fut_idx >= task.futures.size()) continue;
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
#ifdef DEBUG_LEGATE
  assert(mapped_futures.size() <= task.futures.size());
  // The launching code should be packing all Store-backing Futures first.
  if (mapped_futures.size() > 0) {
    uint32_t max_mapped_fut = *mapped_futures.rbegin();
    assert(mapped_futures.size() == max_mapped_fut + 1);
  }
#endif

  // Map future-backed stores
  output.future_locations.resize(mapped_futures.size());
  for (auto& mapping : for_futures) {
    auto fut_idx       = mapping.store().future_index();
    StoreTarget target = mapping.policy.target;
#ifdef LEGATE_NO_FUTURES_ON_FB
    if (target == StoreTarget::FBMEM) target = StoreTarget::ZCMEM;
#endif
    output.future_locations[fut_idx] = machine.get_memory(task.target_proc, target);
  }

  // Map unbound stores
  auto map_unbound_stores = [&](auto& mappings) {
    for (auto& mapping : mappings) {
      auto req_idx                   = mapping.requirement_index();
      output.output_targets[req_idx] = machine.get_memory(task.target_proc, mapping.policy.target);
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
  auto target_memory = machine.get_memory(target_proc, policy.target);

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
      report_failed_mapping(mappable, mapping.requirement_index(), target_memory, redop, footprint);
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
    for (auto req_idx : req_indices)
      report_failed_mapping(mappable, req_idx, target_memory, redop, footprint);
  }
  return true;
}

void BaseMapper::report_failed_mapping(const Legion::Mappable& mappable,
                                       uint32_t index,
                                       Memory target_memory,
                                       Legion::ReductionOpID redop,
                                       size_t footprint)
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

  logger.error("Mapper %s failed to allocate %zd bytes on memory " IDFMT
               " (of kind %s: %s) for %s of %s%s[%s] (UID %lld).\n"
               "This means Legate was unable to reserve ouf of its memory pool the full amount "
               "required for the above operation. Here are some things to try:\n"
               "* Make sure your code is not impeding the garbage collection of Legate-backed "
               "objects, e.g. by storing references in caches, or creating reference cycles.\n"
               "* Ask Legate to reserve more space on the above memory, using the appropriate "
               "--*mem legate flag.\n"
               "* Assign less memory to the eager pool, by reducing --eager-alloc-percentage.\n"
               "* If running on multiple nodes, increase how often distributed garbage collection "
               "runs, by reducing LEGATE_FIELD_REUSE_FREQ (default: 32, warning: may "
               "incur overhead).\n"
               "* Adapt your code to reduce temporary storage requirements, e.g. by breaking up "
               "larger operations into batches.\n"
               "* If the previous steps don't help, and you are confident Legate should be able to "
               "handle your code's working set, please open an issue on Legate's bug tracker.",
               get_mapper_name(),
               footprint,
               target_memory.id,
               Legion::Mapping::Utilities::to_string(target_memory.kind()),
               memory_kinds[target_memory.kind()],
               req_ss.str().c_str(),
               log_mappable(mappable, true /*prefix_only*/).c_str(),
               opname.c_str(),
               provenance.c_str(),
               mappable.get_unique_id());
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

namespace {

using Bandwidth = uint32_t;

// Source instance annotated with the memory bandwidth
struct AnnotatedSourceInstance {
  AnnotatedSourceInstance(Legion::Mapping::PhysicalInstance _instance, Bandwidth _bandwidth)
    : instance(_instance), bandwidth(_bandwidth)
  {
  }
  Legion::Mapping::PhysicalInstance instance;
  Bandwidth bandwidth;
};

void find_source_instance_bandwidth(
  std::vector<AnnotatedSourceInstance>& all_sources,     /* output */
  std::map<Memory, Bandwidth>& source_memory_bandwidths, /* inout */
  const Legion::Mapping::PhysicalInstance& source_instance,
  const Memory& target_memory,
  const Legion::Machine& machine)
{
  Memory source_memory = source_instance.get_location();
  auto finder          = source_memory_bandwidths.find(source_memory);

  uint32_t bandwidth{0};
  if (source_memory_bandwidths.end() == finder) {
    std::vector<Legion::MemoryMemoryAffinity> affinities;
    machine.get_mem_mem_affinity(
      affinities, source_memory, target_memory, false /*not just local affinities*/);
    // affinities being empty means that there's no direct channel between the source
    // and target memories, in which case we assign the smallest bandwidth
    // TODO: Not all multi-hop copies are equal
    if (!affinities.empty()) {
#ifdef DEBUG_LEGATE
      assert(affinities.size() == 1);
#endif
      bandwidth = affinities.front().bandwidth;
    }
    source_memory_bandwidths[source_memory] = bandwidth;
  } else
    bandwidth = finder->second;

  all_sources.emplace_back(source_instance, bandwidth);
}

}  // namespace

void BaseMapper::legate_select_sources(
  const Legion::Mapping::MapperContext ctx,
  const Legion::Mapping::PhysicalInstance& target,
  const std::vector<Legion::Mapping::PhysicalInstance>& sources,
  const std::vector<Legion::Mapping::CollectiveView>& collective_sources,
  std::deque<Legion::Mapping::PhysicalInstance>& ranking)
{
  std::map<Memory, Bandwidth> source_memory_bandwidths;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination.
  // TODO: consider layouts when ranking source to help out the DMA system
  Memory target_memory = target.get_location();
  // fill in a vector of the sources with their bandwidths
  std::vector<AnnotatedSourceInstance> all_sources;

  for (uint32_t idx = 0; idx < sources.size(); idx++)
    find_source_instance_bandwidth(
      all_sources, source_memory_bandwidths, sources[idx], target_memory, legion_machine);

  for (uint32_t idx = 0; idx < collective_sources.size(); idx++) {
    std::vector<Legion::Mapping::PhysicalInstance> source_instances;
    collective_sources[idx].find_instances_nearest_memory(target_memory, source_instances);
#ifdef DEBUG_LEGATE
    // there must exist at least one instance in the collective view
    assert(!source_instances.empty());
#endif
    // we need only first instance if there are several
    find_source_instance_bandwidth(all_sources,
                                   source_memory_bandwidths,
                                   source_instances.front(),
                                   target_memory,
                                   legion_machine);
  }
#ifdef DEBUG_LEGATE
  assert(!all_sources.empty());
#endif
  if (all_sources.size() > 1)
    // Sort source instances by their bandwidths
    std::sort(all_sources.begin(), all_sources.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.bandwidth > rhs.bandwidth;
    });
  // Record all instances from the one of the largest bandwidth to that of the smallest
  for (auto& source : all_sources) ranking.emplace_back(source.instance);
}

void BaseMapper::report_profiling(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Task& task,
                                  const TaskProfilingInfo& input)
{
  // Shouldn't get any profiling feedback currently
  LEGATE_ABORT;
}

Legion::ShardingID BaseMapper::find_mappable_sharding_functor_id(const Legion::Mappable& mappable)
{
  Mappable legate_mappable(&mappable);
  return static_cast<Legion::ShardingID>(legate_mappable.sharding_id());
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_mappable_sharding_functor_id(task);
}

void BaseMapper::map_inline(const Legion::Mapping::MapperContext ctx,
                            const Legion::InlineMapping& inline_op,
                            const MapInlineInput& input,
                            MapInlineOutput& output)
{
  Processor target_proc{Processor::NO_PROC};
  if (machine.has_omps())
    target_proc = machine.omps().front();
  else
    target_proc = machine.cpus().front();

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
  Copy legate_copy(&copy, runtime, ctx);
  auto& machine_desc = legate_copy.machine_desc();
  auto copy_target   = [&]() {
    // If we're mapping an indirect copy and have data resident in GPU memory,
    // map everything to CPU memory, as indirect copies on GPUs are currently
    // extremely slow.
    auto indirect =
      !copy.src_indirect_requirements.empty() || !copy.dst_indirect_requirements.empty();
    auto valid_targets = indirect ? machine_desc.valid_targets_except({TaskTarget::GPU})
                                    : machine_desc.valid_targets();
    // However, if the machine in the scope doesn't have any CPU or OMP as a fallback for
    // indirect copies, we have no choice but using GPUs
    if (valid_targets.empty()) {
#ifdef DEBUG_LEGATE
      assert(indirect);
#endif
      valid_targets = machine_desc.valid_targets();
    }
    return valid_targets.front();
  }();

  auto local_range = machine.slice(copy_target, machine_desc, true);
  Processor target_proc;
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
    auto idx          = linearize(lo, hi, p);
    target_proc       = local_range[idx];
  } else
    target_proc = local_range.first();

  auto store_target = default_store_targets(target_proc.kind()).front();

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
  output.chosen_functor = find_mappable_sharding_functor_id(copy);
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
  if (machine.has_omps())
    target_proc = machine.omps().front();
  else
    target_proc = machine.cpus().front();

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
  output.chosen_functor = find_mappable_sharding_functor_id(partition);
}

void BaseMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Fill& fill,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_mappable_sharding_functor_id(fill);
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
