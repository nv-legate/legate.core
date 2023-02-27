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

#include "legion.h"

#include "core/data/buffer.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/base_mapper.h"
#include "core/runtime/context.h"
#include "core/runtime/runtime.h"
#include "core/utilities/deserializer.h"

#ifdef LEGATE_USE_CUDA
#include "core/cuda/cuda_help.h"
#endif

#include "mappers/logging_wrapper.h"

namespace legate {

LibraryContext::LibraryContext(const std::string& library_name, const ResourceConfig& config)
  : runtime_(Legion::Runtime::get_runtime()), library_name_(library_name)
{
  task_scope_ = ResourceScope(
    runtime_->generate_library_task_ids(library_name.c_str(), config.max_tasks), config.max_tasks);
  mapper_scope_ =
    ResourceScope(runtime_->generate_library_mapper_ids(library_name.c_str(), config.max_mappers),
                  config.max_mappers);
  redop_scope_ = ResourceScope(
    runtime_->generate_library_reduction_ids(library_name.c_str(), config.max_reduction_ops),
    config.max_reduction_ops);
  proj_scope_ = ResourceScope(
    runtime_->generate_library_projection_ids(library_name.c_str(), config.max_projections),
    config.max_projections);
  shard_scope_ = ResourceScope(
    runtime_->generate_library_sharding_ids(library_name.c_str(), config.max_shardings),
    config.max_shardings);
}

const std::string& LibraryContext::get_library_name() const { return library_name_; }

Legion::TaskID LibraryContext::get_task_id(int64_t local_task_id) const
{
  assert(task_scope_.valid());
  return task_scope_.translate(local_task_id);
}

Legion::MapperID LibraryContext::get_mapper_id(int64_t local_mapper_id) const
{
  assert(mapper_scope_.valid());
  return mapper_scope_.translate(local_mapper_id);
}

Legion::ReductionOpID LibraryContext::get_reduction_op_id(int64_t local_redop_id) const
{
  assert(redop_scope_.valid());
  return redop_scope_.translate(local_redop_id);
}

Legion::ProjectionID LibraryContext::get_projection_id(int64_t local_proj_id) const
{
  if (local_proj_id == 0)
    return 0;
  else {
    assert(proj_scope_.valid());
    return proj_scope_.translate(local_proj_id);
  }
}

Legion::ShardingID LibraryContext::get_sharding_id(int64_t local_shard_id) const
{
  assert(shard_scope_.valid());
  return shard_scope_.translate(local_shard_id);
}

int64_t LibraryContext::get_local_task_id(Legion::TaskID task_id) const
{
  assert(task_scope_.valid());
  return task_scope_.invert(task_id);
}

int64_t LibraryContext::get_local_mapper_id(Legion::MapperID mapper_id) const
{
  assert(mapper_scope_.valid());
  return mapper_scope_.invert(mapper_id);
}

int64_t LibraryContext::get_local_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  assert(redop_scope_.valid());
  return redop_scope_.invert(redop_id);
}

int64_t LibraryContext::get_local_projection_id(Legion::ProjectionID proj_id) const
{
  if (proj_id == 0)
    return 0;
  else {
    assert(proj_scope_.valid());
    return proj_scope_.invert(proj_id);
  }
}

int64_t LibraryContext::get_local_sharding_id(Legion::ShardingID shard_id) const
{
  assert(shard_scope_.valid());
  return shard_scope_.invert(shard_id);
}

bool LibraryContext::valid_task_id(Legion::TaskID task_id) const
{
  return task_scope_.in_scope(task_id);
}

bool LibraryContext::valid_mapper_id(Legion::MapperID mapper_id) const
{
  return mapper_scope_.in_scope(mapper_id);
}

bool LibraryContext::valid_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  return redop_scope_.in_scope(redop_id);
}

bool LibraryContext::valid_projection_id(Legion::ProjectionID proj_id) const
{
  return proj_scope_.in_scope(proj_id);
}

bool LibraryContext::valid_sharding_id(Legion::ShardingID shard_id) const
{
  return shard_scope_.in_scope(shard_id);
}

void LibraryContext::register_mapper(std::unique_ptr<mapping::LegateMapper> mapper,
                                     int64_t local_mapper_id) const
{
  auto base_mapper = new legate::mapping::BaseMapper(
    std::move(mapper), runtime_, Realm::Machine::get_machine(), *this);
  Legion::Mapping::Mapper* legion_mapper = base_mapper;
  if (Core::log_mapping_decisions)
    legion_mapper = new Legion::Mapping::LoggingWrapper(base_mapper, &base_mapper->logger);
  runtime_->add_mapper(get_mapper_id(local_mapper_id), legion_mapper);
}

TaskContext::TaskContext(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  : task_(task), regions_(regions), context_(context), runtime_(runtime)
{
  TaskDeserializer dez(task, regions);
  inputs_     = dez.unpack<std::vector<Store>>();
  outputs_    = dez.unpack<std::vector<Store>>();
  reductions_ = dez.unpack<std::vector<Store>>();
  scalars_    = dez.unpack<std::vector<Scalar>>();

  can_raise_exception_ = dez.unpack<bool>();

  bool insert_barrier = false;
  Legion::PhaseBarrier arrival, wait;
  if (task->is_index_space) {
    insert_barrier = dez.unpack<bool>();
    if (insert_barrier) {
      arrival = dez.unpack<Legion::PhaseBarrier>();
      wait    = dez.unpack<Legion::PhaseBarrier>();
    }
    comms_ = dez.unpack<std::vector<comm::Communicator>>();
  }
  // For reduction tree cases, some input stores may be mapped to NO_REGION
  // when the number of subregions isn't a multiple of the chosen radix.
  // To simplify the programming mode, we filter out those "invalid" stores out.
  if (task_->tag == LEGATE_CORE_TREE_REDUCE_TAG) {
    std::vector<Store> inputs;
    for (auto& input : inputs_)
      if (input.valid()) inputs.push_back(std::move(input));
    inputs_.swap(inputs);
  }

  // CUDA drivers < 520 have a bug that causes deadlock under certain circumstances
  // if the application has multiple threads that launch blocking kernels, such as
  // NCCL all-reduce kernels. This barrier prevents such deadlock by making sure
  // all CUDA driver calls from Realm are done before any of the GPU tasks starts
  // making progress.
  if (insert_barrier) {
    arrival.arrive();
    wait.wait();
  }
#ifdef LEGATE_USE_CUDA
  // If the task is running on a GPU and there is at least one scalar store for reduction,
  // we need to wait for all the host-to-device copies for initialization to finish
  if (Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC)
    for (auto& reduction : reductions_)
      if (reduction.is_future()) {
        CHECK_CUDA(cudaDeviceSynchronize());
        break;
      }
#endif
}

bool TaskContext::is_single_task() const { return !task_->is_index_space; }

DomainPoint TaskContext::get_task_index() const { return task_->index_point; }

Domain TaskContext::get_launch_domain() const { return task_->index_domain; }

void TaskContext::make_all_unbound_stores_empty()
{
  for (auto& output : outputs_)
    if (output.is_output_store()) output.make_empty();
}

ReturnValues TaskContext::pack_return_values() const
{
  auto return_values = get_return_values();
  if (can_raise_exception_) {
    ReturnedException exn{};
    return_values.push_back(exn.pack());
  }
  return ReturnValues(std::move(return_values));
}

ReturnValues TaskContext::pack_return_values_with_exception(int32_t index,
                                                            const std::string& error_message) const
{
  auto return_values = get_return_values();
  if (can_raise_exception_) {
    ReturnedException exn(index, error_message);
    return_values.push_back(exn.pack());
  }
  return ReturnValues(std::move(return_values));
}

std::vector<ReturnValue> TaskContext::get_return_values() const
{
  size_t num_unbound_outputs = 0;
  std::vector<ReturnValue> return_values;

  for (auto& output : outputs_) {
    if (!output.is_output_store()) continue;
    return_values.push_back(output.pack_weight());
    ++num_unbound_outputs;
  }
  for (auto& output : outputs_) {
    if (!output.is_future()) continue;
    return_values.push_back(output.pack());
  }
  for (auto& reduction : reductions_) {
    if (!reduction.is_future()) continue;
    return_values.push_back(reduction.pack());
  }

  // If this is a reduction task, we do sanity checks on the invariants
  // the Python code relies on.
  if (task_->tag == LEGATE_CORE_TREE_REDUCE_TAG) {
    if (return_values.size() != 1 || num_unbound_outputs != 1) {
      legate::log_legate.error("Reduction tasks must have only one unbound output and no others");
      LEGATE_ABORT;
    }
  }

  return std::move(return_values);
}

}  // namespace legate
