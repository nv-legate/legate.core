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

#include "core/utilities/debug.h"

#include "core/utilities/dispatch.h"
#include "core/utilities/type_traits.h"

namespace legate {

namespace {  // anonymous

struct print_dense_array_fn {
  template <LegateTypeCode CODE, int DIM>
  std::string operator()(const Store& store)
  {
    using T        = legate_type_of<CODE>;
    Rect<DIM> rect = store.shape<DIM>();
    return print_dense_array(store.read_accessor<T>(rect), rect);
  }
};

}  // namespace

std::string print_dense_array(const Store& store)
{
  assert(store.is_readable());
  return double_dispatch(store.dim(), store.code(), print_dense_array_fn{}, store);
}

}  // namespace legate
