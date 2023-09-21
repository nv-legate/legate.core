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

InvalidTaskIdException::InvalidTaskIdException(const std::string& library_name,
                                               int64_t offending_task_id,
                                               int64_t max_task_id)
{
  std::stringstream ss;
  ss << "Task id " << offending_task_id << " is invalid for library '" << library_name
     << "' (max local task id: " << max_task_id << ")";
  error_message = std::move(ss).str();
}

const char* InvalidTaskIdException::what() const throw() { return error_message.c_str(); }

LibraryContext::LibraryContext(const std::string& library_name,
                               const ResourceConfig& config,
                               std::unique_ptr<mapping::Mapper> mapper)
  : runtime_(Legion::Runtime::get_runtime()),
    library_name_(library_name),
    mapper_(std::move(mapper))
{
  task_scope_ = ResourceIdScope(
    runtime_->generate_library_task_ids(library_name.c_str(), config.max_tasks), config.max_tasks);
  redop_scope_ = ResourceIdScope(
    runtime_->generate_library_reduction_ids(library_name.c_str(), config.max_reduction_ops),
    config.max_reduction_ops);
  proj_scope_ = ResourceIdScope(
    runtime_->generate_library_projection_ids(library_name.c_str(), config.max_projections),
    config.max_projections);
  shard_scope_ = ResourceIdScope(
    runtime_->generate_library_sharding_ids(library_name.c_str(), config.max_shardings),
    config.max_shardings);

  auto base_mapper = new mapping::BaseMapper(mapper_.get(), runtime_->get_mapper_runtime(), this);
  Legion::Mapping::Mapper* legion_mapper = base_mapper;
  if (Core::log_mapping_decisions)
    legion_mapper = new Legion::Mapping::LoggingWrapper(base_mapper, &base_mapper->logger);

  mapper_id_ = runtime_->generate_library_mapper_ids(library_name.c_str(), 1);
  runtime_->add_mapper(mapper_id_, legion_mapper);
}

const std::string& LibraryContext::get_library_name() const { return library_name_; }

Legion::TaskID LibraryContext::get_task_id(int64_t local_task_id) const
{
  assert(task_scope_.valid());
  return task_scope_.translate(local_task_id);
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

void LibraryContext::register_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  auto task_id = get_task_id(local_task_id);
  if (!task_scope_.in_scope(task_id))
    throw InvalidTaskIdException(library_name_, local_task_id, task_scope_.size() - 1);

#ifdef DEBUG_LEGATE
  log_legate.debug() << "[" << library_name_ << "] task " << local_task_id
                     << " (global id: " << task_id << "), " << *task_info;
#endif
  task_info->register_task(task_id);
  tasks_.emplace(std::make_pair(local_task_id, std::move(task_info)));
}

const TaskInfo* LibraryContext::find_task(int64_t local_task_id) const
{
  auto finder = tasks_.find(local_task_id);
  return tasks_.end() == finder ? nullptr : finder->second.get();
}

TaskContext::TaskContext(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  : task_(task), regions_(regions), context_(context), runtime_(runtime)
{
  {
    mapping::MapperDataDeserializer dez(task);
    machine_desc_ = dez.unpack<mapping::MachineDesc>();
  }

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
    if (output.is_unbound_store()) output.bind_empty_data();
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
    if (!output.is_unbound_store()) continue;
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
