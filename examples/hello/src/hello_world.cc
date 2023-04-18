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

#include "hello_world.h"
#include "legate_library.h"

namespace hello {

Legion::Logger logger("legate.hello");

class HelloWorldTask : public Task<HelloWorldTask, HELLO_WORLD> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    std::string message = context.scalars().at(0).value<std::string>();
    std::cout << message << std::endl;
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::HelloWorldTask::register_variants();
}

}  // namespace
