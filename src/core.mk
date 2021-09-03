# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# General source files
GEN_CPU_SRC	= legate_c.cc                 \
							data/scalar.cc              \
							data/store.cc               \
							data/transform.cc           \
							mapping/base_mapper.cc      \
							mapping/core_mapper.cc      \
							mapping/instance_manager.cc \
							mapping/task.cc             \
							runtime/context.cc          \
							runtime/projection.cc       \
							runtime/runtime.cc          \
							runtime/shard.cc            \
							task/task.cc                \
							utilities/deserializer.cc

ifeq ($(strip $(USE_CUDA)),1)
GEN_CPU_SRC	+= gpu/cudalibs.cc
endif

# Header files that we need to have installed for client legate libraries
INSTALL_PATHS = data      \
								mapping   \
								runtime   \
								task      \
								utilities

INSTALL_HEADERS = legate.h                   \
									legate_c.h                 \
									legate_defines.h           \
									legate_preamble.h          \
									data/buffer.h              \
									data/scalar.h              \
									data/scalar.inl            \
									data/store.h               \
									data/store.inl             \
									data/transform.h           \
									mapping/base_mapper.h      \
									mapping/mapping.h          \
									mapping/task.h             \
									mapping/task.inl           \
									runtime/context.h          \
									runtime/runtime.h          \
									task/task.h                \
									utilities/deserializer.h   \
									utilities/deserializer.inl \
									utilities/dispatch.h       \
									utilities/span.h           \
									utilities/type_traits.h    \
									utilities/typedefs.h
