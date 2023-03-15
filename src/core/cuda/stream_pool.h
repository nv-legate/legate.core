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

#include <memory>

#include <cuda_runtime.h>
#include "legion.h"

/**
 * @file
 * @brief Class definition for legate::cuda::StreamPool
 */

namespace legate {
namespace cuda {

/**
 * @ingroup task
 * @brief A simple wrapper around CUDA streams to inject auxiliary features
 *
 * When `LEGATE_SYNC_STREAM_VIEW` is set to 1, every `StreamView` synchronizes the CUDA stream
 * that it wraps when it is destroyed.
 */
struct StreamView {
 public:
  /**
   * @brief Creates a `StreamView` with a raw CUDA stream
   *
   * @param stream Raw CUDA stream to wrap
   */
  StreamView(cudaStream_t stream) : valid_(true), stream_(stream) {}
  ~StreamView();

 public:
  StreamView(const StreamView&)            = delete;
  StreamView& operator=(const StreamView&) = delete;

 public:
  StreamView(StreamView&&);
  StreamView& operator=(StreamView&&);

 public:
  /**
   * @brief Unwraps the raw CUDA stream
   *
   * @return Raw CUDA stream wrapped by the `StreamView`
   */
  operator cudaStream_t() const { return stream_; }

 private:
  bool valid_;
  cudaStream_t stream_;
};

/**
 * @brief A stream pool
 */
struct StreamPool {
 public:
  StreamPool() {}
  ~StreamPool();

 public:
  /**
   * @brief Returns a `StreamView` in the pool
   *
   * @return A `StreamView` object. Currently, all stream views returned from this pool are backed
   * by the same CUDA stream.
   */
  StreamView get_stream();

 public:
  /**
   * @brief Returns a singleton stream pool
   *
   * The stream pool is alive throughout the program execution.
   *
   * @return A `StreamPool` object
   */
  static StreamPool& get_stream_pool();

 private:
  // For now we keep only one stream in the pool
  // TODO: If this ever changes, the use of non-stream-ordered `DeferredBuffer`s
  // in `core/data/buffer.h` will no longer be safe.
  std::unique_ptr<cudaStream_t> cached_stream_{nullptr};
};

}  // namespace cuda
}  // namespace legate
