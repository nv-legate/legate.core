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

#include "legion.h"

#include "core/utilities/typedefs.h"
#include "core/runtime/context.h"
#include <unordered_map>
namespace legate {

using LegateVariantImpl = void (*)(TaskContext&);

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

class Core {
 public:
  static void parse_config(void);
  static void shutdown(void);
  static std::unordered_map<int64_t, LegateVariantImpl> cpuDescriptors; 
  static std::unordered_map<int64_t, LegateVariantImpl> gpuDescriptors; 
  static std::vector<std::pair<int64_t, LegateVariantImpl> > opIDs;
  static std::vector<std::pair<int64_t, LegateVariantImpl> > gpuOpIDs;

 public:
  // Configuration settings
  static bool show_progress;
#ifdef LEGATE_USE_CUDA
 public:
  static cublasContext* get_cublas(void);
#endif
};

}  // namespace legate
