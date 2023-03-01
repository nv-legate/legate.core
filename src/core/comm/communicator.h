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

/**
 * @file
 * @brief Class definition for legate::comm::Communicator
 */

namespace legate {
namespace comm {

/**
 * @ingroup task
 * @brief A thin wrapper class for communicators stored in futures. This class only provides
 * a tempalte method to retrieve the communicator handle and the client is expected to pass
 * the right handle type.
 *
 * The following is the list of handle types for communicators supported in Legate:
 *
 *   - NCCL: ncclComm_t*
 *   - CPU communicator in Legate: legate::comm::coll::CollComm*
 */
class Communicator {
 public:
  Communicator() {}
  Communicator(Legion::Future future) : future_(future) {}

 public:
  Communicator(const Communicator&)            = default;
  Communicator& operator=(const Communicator&) = default;

 public:
  /**
   * @brief Returns the communicator stored in the wrapper
   *
   * @tparam T The type of communicator handle to get (see valid types above)
   *
   * @return A communicator
   */
  template <typename T>
  T get() const
  {
    return future_.get_result<T>();
  }

 private:
  Legion::Future future_{};
};

}  // namespace comm
}  // namespace legate
