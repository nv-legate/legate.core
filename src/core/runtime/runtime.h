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

#include "legion.h"

#include <memory>

#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"

/** @defgroup runtime Runtime and library contexts
 */

namespace legate {

class LibraryContext;

namespace mapping {

class Mapper;

}  // namespace mapping

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

/**
 * @ingroup runtime
 * @brief A utility class that collects static members shared by all Legate libraries
 */
struct Core {
 public:
  static void parse_config(void);
  static void shutdown(void);
  static void show_progress(const Legion::Task* task,
                            Legion::Context ctx,
                            Legion::Runtime* runtime);
  static void report_unexpected_exception(const Legion::Task* task, const legate::TaskException& e);

 public:
  /**
   * @brief Type signature for registration callbacks
   */
  using RegistrationCallback = void (*)();

  /**
   * @brief Performs a registration callback. Libraries must perform
   * registration of tasks and other components through this function.
   *
   * @tparam CALLBACK Registration callback to perform
   */
  template <RegistrationCallback CALLBACK>
  static void perform_registration();

 public:
  // Configuration settings
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool has_socket_mem;
  static bool warmup_nccl;
};

class Runtime {
 private:
  Runtime();
  ~Runtime();

 public:
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  LibraryContext* create_library(const std::string& library_name,
                                 const ResourceConfig& config            = ResourceConfig{},
                                 std::unique_ptr<mapping::Mapper> mapper = nullptr);

 public:
  uint32_t get_type_uid();
  void record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id);
  int32_t find_reduction_operator(int32_t type_uid, int32_t op_kind) const;

 public:
  static Runtime* get_runtime();

 private:
  std::map<std::string, LibraryContext*> libraries_{};

 private:
  uint32_t next_type_uid_;
  std::map<std::pair<int32_t, int32_t>, int32_t> reduction_ops_{};
};

}  // namespace legate

#include "core/runtime/runtime.inl"
