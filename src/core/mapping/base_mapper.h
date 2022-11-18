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

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "legion.h"

#include "core/data/scalar.h"
#include "core/mapping/mapping.h"
#include "core/runtime/context.h"
#include "core/utilities/typedefs.h"

namespace legate {
namespace mapping {

class InstanceManager;
class Machine;

class BaseMapper : public Legion::Mapping::Mapper, public LegateMapper {
 public:
  BaseMapper(Legion::Runtime* rt, Legion::Machine machine, const LibraryContext& context);
  virtual ~BaseMapper(void);

 private:
  BaseMapper(const BaseMapper& rhs)            = delete;
  BaseMapper& operator=(const BaseMapper& rhs) = delete;

 protected:
  // Start-up methods
  std::string create_name(Legion::AddressSpace node) const;
  std::string create_logger_name() const;

 public:
  virtual const char* get_mapper_name(void) const override;
  virtual Legion::Mapping::Mapper::MapperSyncModel get_mapper_sync_model(void) const override;
  virtual bool request_valid_instances(void) const override { return false; }

 public:  // Task mapping calls
  virtual void select_task_options(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   TaskOptions& output) override;
  virtual void premap_task(const Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           const PremapTaskInput& input,
                           PremapTaskOutput& output) override;
  virtual void slice_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output) override;
  virtual void map_task(const Legion::Mapping::MapperContext ctx,
                        const Legion::Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output) override;
  virtual void map_replicate_task(const Legion::Mapping::MapperContext ctx,
                                  const Legion::Task& task,
                                  const MapTaskInput& input,
                                  const MapTaskOutput& default_output,
                                  MapReplicateTaskOutput& output) override;
  virtual void select_task_variant(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   const SelectVariantInput& input,
                                   SelectVariantOutput& output) override;
  virtual void postmap_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const PostMapInput& input,
                            PostMapOutput& output) override;
  virtual void select_task_sources(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task& task,
                                   const SelectTaskSrcInput& input,
                                   SelectTaskSrcOutput& output) override;
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task& task,
                         SpeculativeOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Task& task,
                                const TaskProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Inline mapping calls
  virtual void map_inline(const Legion::Mapping::MapperContext ctx,
                          const Legion::InlineMapping& inline_op,
                          const MapInlineInput& input,
                          MapInlineOutput& output) override;
  virtual void select_inline_sources(const Legion::Mapping::MapperContext ctx,
                                     const Legion::InlineMapping& inline_op,
                                     const SelectInlineSrcInput& input,
                                     SelectInlineSrcOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::InlineMapping& inline_op,
                                const InlineProfilingInfo& input) override;

 public:  // Copy mapping calls
  virtual void map_copy(const Legion::Mapping::MapperContext ctx,
                        const Legion::Copy& copy,
                        const MapCopyInput& input,
                        MapCopyOutput& output) override;
  virtual void select_copy_sources(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Copy& copy,
                                   const SelectCopySrcInput& input,
                                   SelectCopySrcOutput& output) override;
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Copy& copy,
                         SpeculativeOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Copy& copy,
                                const CopyProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Copy& copy,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Close mapping calls
  virtual void select_close_sources(const Legion::Mapping::MapperContext ctx,
                                    const Legion::Close& close,
                                    const SelectCloseSrcInput& input,
                                    SelectCloseSrcOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Close& close,
                                const CloseProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Close& close,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Acquire mapping calls
  virtual void map_acquire(const Legion::Mapping::MapperContext ctx,
                           const Legion::Acquire& acquire,
                           const MapAcquireInput& input,
                           MapAcquireOutput& output) override;
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Acquire& acquire,
                         SpeculativeOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Acquire& acquire,
                                const AcquireProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Acquire& acquire,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Release mapping calls
  virtual void map_release(const Legion::Mapping::MapperContext ctx,
                           const Legion::Release& release,
                           const MapReleaseInput& input,
                           MapReleaseOutput& output) override;
  virtual void select_release_sources(const Legion::Mapping::MapperContext ctx,
                                      const Legion::Release& release,
                                      const SelectReleaseSrcInput& input,
                                      SelectReleaseSrcOutput& output) override;
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Release& release,
                         SpeculativeOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Release& release,
                                const ReleaseProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Release& release,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Partition mapping calls
  virtual void select_partition_projection(const Legion::Mapping::MapperContext ctx,
                                           const Legion::Partition& partition,
                                           const SelectPartitionProjectionInput& input,
                                           SelectPartitionProjectionOutput& output) override;
  virtual void map_partition(const Legion::Mapping::MapperContext ctx,
                             const Legion::Partition& partition,
                             const MapPartitionInput& input,
                             MapPartitionOutput& output) override;
  virtual void select_partition_sources(const Legion::Mapping::MapperContext ctx,
                                        const Legion::Partition& partition,
                                        const SelectPartitionSrcInput& input,
                                        SelectPartitionSrcOutput& output) override;
  virtual void report_profiling(const Legion::Mapping::MapperContext ctx,
                                const Legion::Partition& partition,
                                const PartitionProfilingInfo& input) override;
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Partition& partition,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Fill mapper calls
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Fill& fill,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output) override;

 public:  // Task execution mapping calls
  virtual void configure_context(const Legion::Mapping::MapperContext ctx,
                                 const Legion::Task& task,
                                 ContextConfigOutput& output) override;
  virtual void select_tunable_value(const Legion::Mapping::MapperContext ctx,
                                    const Legion::Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output) override;

 public:  // Must epoch mapping
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::MustEpoch& epoch,
                                       const SelectShardingFunctorInput& input,
                                       MustEpochShardingFunctorOutput& output) override;
  virtual void memoize_operation(const Legion::Mapping::MapperContext ctx,
                                 const Legion::Mappable& mappable,
                                 const MemoizeInput& input,
                                 MemoizeOutput& output) override;
  virtual void map_must_epoch(const Legion::Mapping::MapperContext ctx,
                              const MapMustEpochInput& input,
                              MapMustEpochOutput& output) override;

 public:  // Dataflow graph mapping
  virtual void map_dataflow_graph(const Legion::Mapping::MapperContext ctx,
                                  const MapDataflowGraphInput& input,
                                  MapDataflowGraphOutput& output) override;

 public:  // Mapping control and stealing
  virtual void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                                   const SelectMappingInput& input,
                                   SelectMappingOutput& output) override;
  virtual void select_steal_targets(const Legion::Mapping::MapperContext ctx,
                                    const SelectStealingInput& input,
                                    SelectStealingOutput& output) override;
  virtual void permit_steal_request(const Legion::Mapping::MapperContext ctx,
                                    const StealRequestInput& intput,
                                    StealRequestOutput& output) override;

 public:  // handling
  virtual void handle_message(const Legion::Mapping::MapperContext ctx,
                              const MapperMessage& message) override;
  virtual void handle_task_result(const Legion::Mapping::MapperContext ctx,
                                  const MapperTaskResult& result) override;

 protected:
  using OutputMap =
    std::map<const Legion::RegionRequirement*, std::vector<Legion::Mapping::PhysicalInstance>*>;
  void map_legate_stores(const Legion::Mapping::MapperContext ctx,
                         const Legion::Mappable& mappable,
                         std::vector<StoreMapping>& mappings,
                         Legion::Processor target_proc,
                         OutputMap& output_map);
  void tighten_write_policies(const Legion::Mappable& mappable,
                              std::vector<StoreMapping>& mappings);
  bool map_legate_store(const Legion::Mapping::MapperContext ctx,
                        const Legion::Mappable& mappable,
                        const StoreMapping& mapping,
                        const std::set<const Legion::RegionRequirement*>& reqs,
                        Legion::Processor target_proc,
                        Legion::Mapping::PhysicalInstance& result,
                        bool can_fail);
  void report_failed_mapping(const Legion::Mappable& mappable,
                             unsigned index,
                             Legion::Memory target_memory,
                             Legion::ReductionOpID redop);
  void legate_select_sources(const Legion::Mapping::MapperContext ctx,
                             const Legion::Mapping::PhysicalInstance& target,
                             const std::vector<Legion::Mapping::PhysicalInstance>& sources,
                             std::deque<Legion::Mapping::PhysicalInstance>& ranking);
  Legion::ShardingID find_mappable_sharding_functor_id(const Legion::Mappable& mappable);

 protected:
  bool has_variant(const Legion::Mapping::MapperContext ctx,
                   const Legion::Task& task,
                   TaskTarget target);
  std::optional<Legion::VariantID> find_variant(const Legion::Mapping::MapperContext ctx,
                                                const Legion::Task& task,
                                                Legion::Processor::Kind kind);

 protected:
  void slice_auto_task(const Legion::Mapping::MapperContext ctx,
                       const Legion::Task& task,
                       const Span<const Legion::Processor>& avail_procs,
                       uint32_t size,
                       uint32_t offset,
                       const SliceTaskInput& input,
                       SliceTaskOutput& output);

 protected:
  Legion::ShardingID find_sharding_functor_by_key_store_projection(
    const std::vector<Legion::RegionRequirement>& requirements);

 protected:
  static inline bool physical_sort_func(
    const std::pair<Legion::Mapping::PhysicalInstance, unsigned>& left,
    const std::pair<Legion::Mapping::PhysicalInstance, unsigned>& right)
  {
    return (left.second < right.second);
  }
  // NumPyOpCode decode_task_id(Legion::TaskID tid);

 public:
  Legion::Runtime* const legion_runtime;
  const Legion::Machine legion_machine;
  const LibraryContext context;
  Legion::Logger logger;

 private:
  std::string mapper_name;

 protected:
  using VariantCacheKey = std::pair<Legion::TaskID, Legion::Processor::Kind>;
  std::map<VariantCacheKey, std::optional<Legion::VariantID>> variants;

 protected:
  InstanceManager* local_instances;
  std::unique_ptr<Machine> machine;
};

}  // namespace mapping
}  // namespace legate
