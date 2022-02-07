# Copyright 2021-2022 NVIDIA Corporation
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
GEN_CPU_SRC	= core/legate_c.cc                 \
							core/data/scalar.cc              \
							core/data/store.cc               \
							core/data/transform.cc           \
							core/mapping/base_mapper.cc      \
							core/mapping/core_mapper.cc      \
							core/mapping/instance_manager.cc \
							core/mapping/mapping.cc          \
							core/mapping/task.cc             \
							core/runtime/context.cc          \
							core/runtime/projection.cc       \
							core/runtime/runtime.cc          \
							core/runtime/shard.cc            \
							core/task/return.cc              \
							core/task/task.cc                \
							core/utilities/debug.cc          \
							core/utilities/deserializer.cc   \
							core/utilities/machine.cc        \
							core/utilities/linearize.cc

# Header files that we need to have installed for client legate libraries
INSTALL_HEADERS = legate.h                        \
									legate_defines.h                \
									legate_preamble.h               \
									core/legate_c.h                 \
									core/data/buffer.h              \
									core/data/scalar.h              \
									core/data/scalar.inl            \
									core/data/store.h               \
									core/data/store.inl             \
									core/data/transform.h           \
									core/mapping/base_mapper.h      \
									core/mapping/mapping.h          \
									core/mapping/task.h             \
									core/mapping/task.inl           \
									core/runtime/context.h          \
									core/runtime/runtime.h          \
									core/task/return.h              \
									core/task/task.h                \
									core/utilities/debug.h          \
									core/utilities/deserializer.h   \
									core/utilities/deserializer.inl \
									core/utilities/dispatch.h       \
									core/utilities/machine.h        \
									core/utilities/span.h           \
									core/utilities/type_traits.h    \
									core/utilities/typedefs.h
