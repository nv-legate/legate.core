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

#include <exception>

/**
 * @file
 * @brief Class definition for legate::TaskException
 */

namespace legate {

/**
 * @ingroup task
 * @brief An exception class used in cross language exception handling
 *
 * Any client that needs to catch a C++ exception during task execution and have it rethrown
 * on the launcher side should wrap that C++ exception with a `TaskException`. In case the
 * task can raise more than one type of exception, they are distinguished by integer ids;
 * the launcher is responsible for enumerating a list of all exceptions that can be raised
 * and the integer ids are positions in that list.
 */
class TaskException : public std::exception {
 public:
  /**
   * @brief Constructs a `TaskException` object with an exception id and an error message.
   * The id must be a valid index for the list of exceptions declared by the launcher.
   *
   * @param index Exception id
   * @param error_message Error message
   */
  TaskException(int32_t index, const std::string& error_message)
    : index_(index), error_message_(error_message)
  {
  }

  /**
   * @brief Constructs a `TaskException` object with an error message. The exception id
   * is set to 0.
   *
   * @param error_message Error message
   */
  TaskException(const std::string& error_message) : index_(0), error_message_(error_message) {}

 public:
  virtual const char* what() const throw() { return error_message_.c_str(); }

 public:
  /**
   * @brief Returns the exception id
   *
   * @return The exception id
   */
  int32_t index() const { return index_; }
  /**
   * @brief Returns the error message
   *
   * @return The error message
   */
  const std::string& error_message() const { return error_message_; }

 private:
  int32_t index_{-1};
  std::string error_message_;
};

}  // namespace legate
