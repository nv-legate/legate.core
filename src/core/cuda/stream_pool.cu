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

#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/runtime/runtime.h"

namespace legate {
namespace cuda {

StreamView::~StreamView()
{
  if (valid_ && Core::synchronize_stream_view) {
#ifdef DEBUG_LEGATE
    CHECK_CUDA_STREAM(stream_);
#else
    CHECK_CUDA(cudaStreamSynchronize(stream_));
#endif
  }
}

StreamView::StreamView(StreamView&& rhs) : valid_(rhs.valid_), stream_(rhs.stream_)
{
  rhs.valid_ = false;
}

StreamView& StreamView::operator=(StreamView&& rhs)
{
  valid_     = rhs.valid_;
  stream_    = rhs.stream_;
  rhs.valid_ = false;
  return *this;
}

StreamPool::~StreamPool()
{
  if (cached_stream_ != nullptr) CHECK_CUDA(cudaStreamDestroy(*cached_stream_));
}

StreamView StreamPool::get_stream()
{
  if (nullptr == cached_stream_) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cached_stream_ = std::make_unique<cudaStream_t>(stream);
  }
  return StreamView(*cached_stream_);
}

/*static*/ StreamPool& StreamPool::get_stream_pool()
{
  static StreamPool pools[LEGION_MAX_NUM_PROCS];
  const auto proc = Legion::Processor::get_executing_processor();
  auto proc_id    = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return pools[proc_id];
}

}  // namespace cuda
}  // namespace legate
