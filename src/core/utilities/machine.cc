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

#include "core/runtime/runtime.h"
#include "legate_defines.h"

namespace legate {

Memory::Kind find_memory_kind_for_executing_processor(bool host_accessible)
{
  auto proc = Processor::get_executing_processor();
  switch (proc.kind()) {
    case Processor::Kind::LOC_PROC: {
      return Memory::Kind::SYSTEM_MEM;
    }
    case Processor::Kind::TOC_PROC: {
      return host_accessible ? Memory::Kind::Z_COPY_MEM : Memory::Kind::GPU_FB_MEM;
    }
    case Processor::Kind::OMP_PROC: {
      return Core::has_socket_mem ? Memory::Kind::SOCKET_MEM : Memory::Kind::SYSTEM_MEM;
    }
    default: break;
  }
  LEGATE_ABORT;
  return Memory::Kind::SYSTEM_MEM;
}

}  // namespace legate
