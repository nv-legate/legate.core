/* Copyright 2022 NVIDIA Corporation
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

#include <stddef.h>

namespace legate {

// We're going to allow for each task to use only up to 341 scalar output stores
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

struct VariantOptions {
  bool leaf{true};
  bool inner{false};
  bool idempotent{false};
  bool concurrent{false};
  size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  VariantOptions& with_leaf(bool leaf);
  VariantOptions& with_inner(bool inner);
  VariantOptions& with_idempotent(bool idempotent);
  VariantOptions& with_concurrent(bool concurrent);
  VariantOptions& with_return_size(size_t return_size);
};

}  // namespace legate
