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

#include "core/comm/comm.h"
#include "core/mapping/core_mapper.h"
#include "core/mapping/default_mapper.h"
#include "core/runtime/context.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/task/exception.h"
#include "core/task/task.h"
#include "core/type/type_info.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/machine.h"
#include "core/utilities/nvtx_help.h"
#include "legate.h"

namespace legate {

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress_requested = false;

/*static*/ bool Core::use_empty_task = false;

/*static*/ bool Core::synchronize_stream_view = false;

/*static*/ bool Core::log_mapping_decisions = false;

/*static*/ bool Core::has_socket_mem = false;

/*static*/ bool Core::warmup_nccl = false;

/*static*/ void Core::parse_config(void)
{
#ifndef LEGATE_USE_CUDA
  const char* need_cuda = getenv("LEGATE_NEED_CUDA");
  if (need_cuda != nullptr) {
    fprintf(stderr,
            "Legate was run with GPUs but was not built with GPU support. "
            "Please install Legate again with the \"--cuda\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_OPENMP
  const char* need_openmp = getenv("LEGATE_NEED_OPENMP");
  if (need_openmp != nullptr) {
    fprintf(stderr,
            "Legate was run with OpenMP processors but was not built with "
            "OpenMP support. Please install Legate again with the \"--openmp\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_NETWORK
  const char* need_network = getenv("LEGATE_NEED_NETWORK");
  if (need_network != nullptr) {
    fprintf(stderr,
            "Legate was run on multiple nodes but was not built with networking "
            "support. Please install Legate again with \"--network\".\n");
    exit(1);
  }
#endif
  auto parse_variable = [](const char* variable, bool& result) {
    const char* value = getenv(variable);
    if (value != nullptr && atoi(value) > 0) result = true;
  };

  parse_variable("LEGATE_SHOW_PROGRESS", show_progress_requested);
  parse_variable("LEGATE_EMPTY_TASK", use_empty_task);
  parse_variable("LEGATE_SYNC_STREAM_VIEW", synchronize_stream_view);
  parse_variable("LEGATE_LOG_MAPPING", log_mapping_decisions);
  parse_variable("LEGATE_WARMUP_NCCL", warmup_nccl);
}

static void extract_scalar_task(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  Core::show_progress(task, legion_context, runtime);

  TaskContext context(task, *regions, legion_context, runtime);
  auto idx            = context.scalars()[0].value<int32_t>();
  auto value_and_size = ReturnValues::extract(task->futures[0], idx);

  // Legion postamble
  value_and_size.finalize(legion_context);
}

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

/*static*/ void Core::show_progress(const Legion::Task* task,
                                    Legion::Context ctx,
                                    Legion::Runtime* runtime)
{
  if (!Core::show_progress_requested) return;
  const auto exec_proc     = runtime->get_executing_processor(ctx);
  const auto proc_kind_str = (exec_proc.kind() == Processor::LOC_PROC)   ? "CPU"
                             : (exec_proc.kind() == Processor::TOC_PROC) ? "GPU"
                                                                         : "OpenMP";

  std::stringstream point_str;
  const auto& point = task->index_point;
  point_str << point[0];
  for (int32_t dim = 1; dim < point.dim; ++dim) point_str << "," << point[dim];

  log_legate.print("%s %s task [%s], pt = (%s), proc = " IDFMT,
                   task->get_task_name(),
                   proc_kind_str,
                   task->get_provenance_string().c_str(),
                   point_str.str().c_str(),
                   exec_proc.id);
}

/*static*/ void Core::report_unexpected_exception(const Legion::Task* task,
                                                  const legate::TaskException& e)
{
  log_legate.error(
    "Task %s threw an exception \"%s\", but the task did not declare any exception. "
    "Please specify a Python exception that you want this exception to be re-thrown with "
    "using 'throws_exception'.",
    task->get_task_name(),
    e.error_message().c_str());
  LEGATE_ABORT;
}

namespace {

constexpr uint32_t CUSTOM_TYPE_UID_BASE = 0x10000;

}  // namespace

Runtime::Runtime() : next_type_uid_(CUSTOM_TYPE_UID_BASE) {}

Runtime::~Runtime()
{
  for (auto& [_, context] : libraries_) delete context;
}

LibraryContext* Runtime::find_library(const std::string& library_name,
                                      bool can_fail /*=false*/) const
{
  auto finder = libraries_.find(library_name);
  if (libraries_.end() == finder) {
    if (!can_fail) {
      log_legate.error("Library %s does not exist", library_name.c_str());
      LEGATE_ABORT;
    } else
      return nullptr;
  }
  return finder->second;
}

LibraryContext* Runtime::create_library(const std::string& library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    log_legate.error("Library %s already exists", library_name.c_str());
    LEGATE_ABORT;
  }

  log_legate.debug("Library %s is created", library_name.c_str());
  if (nullptr == mapper) mapper = std::make_unique<mapping::DefaultMapper>();
  auto context             = new LibraryContext(library_name, config, std::move(mapper));
  libraries_[library_name] = context;
  return context;
}

uint32_t Runtime::get_type_uid() { return next_type_uid_++; }

void Runtime::record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id)
{
#ifdef DEBUG_LEGATE
  log_legate.debug("Record reduction op (type_uid: %d, op_kind: %d, legion_op_id: %d)",
                   type_uid,
                   op_kind,
                   legion_op_id);
#endif
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (finder != reduction_ops_.end()) {
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " already exists for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
  reduction_ops_.emplace(std::make_pair(key, legion_op_id));
}

int32_t Runtime::find_reduction_operator(int32_t type_uid, int32_t op_kind) const
{
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (reduction_ops_.end() == finder) {
#ifdef DEBUG_LEGATE
    log_legate.debug("Can't find reduction op (type_uid: %d, op_kind: %d)", type_uid, op_kind);
#endif
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " does not exist for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
#ifdef DEBUG_LEGATE
  log_legate.debug(
    "Found reduction op %d (type_uid: %d, op_kind: %d)", finder->second, type_uid, op_kind);
#endif
  return finder->second;
}

/*static*/ Runtime* Runtime::get_runtime()
{
  static Runtime runtime;
  return &runtime;
}

void register_legate_core_tasks(Legion::Machine machine,
                                Legion::Runtime* runtime,
                                LibraryContext* context)
{
  auto task_info                       = std::make_unique<TaskInfo>("core::extract_scalar");
  auto register_extract_scalar_variant = [&](auto variant_id) {
    Legion::CodeDescriptor desc(extract_scalar_task);
    // TODO: We could support Legion & Realm calling convensions so we don't pass nullptr here
    task_info->add_variant(variant_id, nullptr, desc, VariantOptions{});
  };
  register_extract_scalar_variant(LEGATE_CPU_VARIANT);
#ifdef LEGATE_USE_CUDA
  register_extract_scalar_variant(LEGATE_GPU_VARIANT);
#endif
#ifdef LEGATE_USE_OPENMP
  register_extract_scalar_variant(LEGATE_OMP_VARIANT);
#endif
  context->register_task(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, std::move(task_info));

  comm::register_tasks(machine, runtime, context);
}

#define BUILTIN_REDOP_ID(OP, TYPE_CODE) \
  (LEGION_REDOP_BASE + (OP) * LEGION_TYPE_TOTAL + (static_cast<int32_t>(TYPE_CODE)))

#define RECORD(OP, TYPE_CODE) \
  PrimitiveType(TYPE_CODE).record_reduction_operator(OP, BUILTIN_REDOP_ID(OP, TYPE_CODE));

#define RECORD_INT(OP)           \
  RECORD(OP, Type::Code::BOOL)   \
  RECORD(OP, Type::Code::INT8)   \
  RECORD(OP, Type::Code::INT16)  \
  RECORD(OP, Type::Code::INT32)  \
  RECORD(OP, Type::Code::INT64)  \
  RECORD(OP, Type::Code::UINT8)  \
  RECORD(OP, Type::Code::UINT16) \
  RECORD(OP, Type::Code::UINT32) \
  RECORD(OP, Type::Code::UINT64)

#define RECORD_FLOAT(OP)          \
  RECORD(OP, Type::Code::FLOAT16) \
  RECORD(OP, Type::Code::FLOAT32) \
  RECORD(OP, Type::Code::FLOAT64)

#define RECORD_COMPLEX(OP) RECORD(OP, Type::Code::COMPLEX64)

#define RECORD_ALL(OP) \
  RECORD_INT(OP)       \
  RECORD_FLOAT(OP)     \
  RECORD_COMPLEX(OP)

void register_builtin_reduction_ops()
{
  RECORD_ALL(ADD_LT)
  RECORD(ADD_LT, Type::Code::COMPLEX128)
  RECORD_ALL(SUB_LT)
  RECORD_ALL(MUL_LT)
  RECORD_ALL(DIV_LT)

  RECORD_INT(MAX_LT)
  RECORD_FLOAT(MAX_LT)

  RECORD_INT(MIN_LT)
  RECORD_FLOAT(MIN_LT)

  RECORD_INT(OR_LT)
  RECORD_INT(AND_LT)
  RECORD_INT(XOR_LT)
}

extern void register_exception_reduction_op(Legion::Runtime* runtime,
                                            const LibraryContext* context);

/*static*/ void core_registration_callback(Legion::Machine machine,
                                           Legion::Runtime* runtime,
                                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings     = LEGATE_CORE_MAX_FUNCTOR_ID;
  config.max_reduction_ops = LEGATE_CORE_MAX_REDUCTION_OP_ID;
  auto core_lib            = Runtime::get_runtime()->create_library(
    core_library_name, config, mapping::create_core_mapper());

  register_legate_core_tasks(machine, runtime, core_lib);

  register_builtin_reduction_ops();

  register_exception_reduction_op(runtime, core_lib);

  register_legate_core_projection_functors(runtime, core_lib);

  register_legate_core_sharding_functors(runtime, core_lib);

  auto fut = runtime->select_tunable_value(
    Legion::Runtime::get_context(), LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM, core_lib->get_mapper_id());
  Core::has_socket_mem = fut.get_result<bool>();
}

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::core_registration_callback,
                                                 true /*global*/);
}
}
