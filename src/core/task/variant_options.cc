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

#include "core/task/variant_options.h"

namespace legate {

VariantOptions& VariantOptions::with_leaf(bool _leaf)
{
  leaf = _leaf;
  return *this;
}

VariantOptions& VariantOptions::with_inner(bool _inner)
{
  inner = _inner;
  return *this;
}

VariantOptions& VariantOptions::with_idempotent(bool _idempotent)
{
  idempotent = _idempotent;
  return *this;
}

VariantOptions& VariantOptions::with_concurrent(bool _concurrent)
{
  concurrent = _concurrent;
  return *this;
}

VariantOptions& VariantOptions::with_return_size(size_t _return_size)
{
  return_size = _return_size;
  return *this;
}

void VariantOptions::populate_registrar(Legion::TaskVariantRegistrar& registrar)
{
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
  registrar.set_concurrent(concurrent);
}

std::ostream& operator<<(std::ostream& os, const VariantOptions& options)
{
  std::stringstream ss;
  ss << "(";
  if (options.leaf) ss << "leaf,";
  if (options.concurrent) ss << "concurrent,";
  ss << options.return_size << ")";
  os << std::move(ss).str();
  return os;
}

}  // namespace legate
