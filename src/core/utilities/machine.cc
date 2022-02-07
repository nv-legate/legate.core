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

#include "core/utilities/machine.h"

using namespace Legion;

namespace legate {

Memory::Kind find_memory_kind_for_executing_processor()
{
  auto proc = Processor::get_executing_processor();
  return proc.kind() == Processor::Kind::TOC_PROC ? Memory::Kind::Z_COPY_MEM
                                                  : Memory::Kind::SYSTEM_MEM;
}

}  // namespace legate
