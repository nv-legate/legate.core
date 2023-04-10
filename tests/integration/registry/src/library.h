/* Copyright 2023 NVIDIA Corporation
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

#include "legate.h"

namespace rg {

extern Legion::Logger log_registry;

struct Registry {
  static legate::TaskRegistrar &get_registrar();
};

template <typename T, int ID> struct Task : public legate::LegateTask<T> {
  using Registrar = Registry;
  static constexpr int TASK_ID = ID;
};

} // namespace rg
