#=============================================================================
# Copyright 2022 NVIDIA Corporation
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
#=============================================================================

option(BUILD_SHARED_LIBS "Build legate.core shared libraries" ON)

function(set_or_default var_name var_env)
  list(LENGTH ARGN num_extra_args)
  if(num_extra_args GREATER 0)
    list(GET ARGN 0 var_default)
  endif()
  if(DEFINED ${var_name})
    message(VERBOSE "legate.core: ${var_name}=${${var_name}}")
  elseif(DEFINED ENV{${var_env}})
    set(${var_name} $ENV{${var_env}} PARENT_SCOPE)
    message(VERBOSE "legate.core: ${var_name}=$ENV{${var_env}} (from envvar '${var_env}')")
  elseif(DEFINED var_default)
    set(${var_name} ${var_default} PARENT_SCOPE)
    message(VERBOSE "legate.core: ${var_name}=${var_default} (from default value)")
  else()
    message(VERBOSE "legate.core: not setting ${var_name}")
  endif()
endfunction()

# Initialize these vars from the CLI, then fallback to an envvar or a default value.
set_or_default(Legion_SPY USE_SPY OFF)
set_or_default(Legion_USE_LLVM USE_LLVM OFF)
set_or_default(Legion_USE_CUDA USE_CUDA OFF)
set_or_default(Legion_USE_HDF5 USE_HDF OFF)
set_or_default(Legion_NETWORKS NETWORKS "")
set_or_default(Legion_USE_OpenMP USE_OPENMP OFF)
set_or_default(Legion_BOUNDS_CHECKS CHECK_BOUNDS OFF)

option(Legion_SPY "Enable detailed logging for Legion Spy" OFF)
option(Legion_USE_LLVM "Use LLVM JIT operations" OFF)
option(Legion_USE_HDF5 "Enable support for HDF5" OFF)
option(Legion_USE_CUDA "Enable Legion support for the CUDA runtime" OFF)
option(Legion_NETWORKS "Networking backends to use (semicolon-separated)" "")
option(Legion_USE_OpenMP "Use OpenMP" OFF)
option(Legion_USE_Python "Use Python" OFF)
option(Legion_BOUNDS_CHECKS "Enable bounds checking in Legion accessors" OFF)

if("${Legion_NETWORKS}" MATCHES ".*gasnet(1|ex).*")
  set_or_default(GASNet_ROOT_DIR GASNET)
  set_or_default(GASNet_CONDUIT CONDUIT "mpi")

  if(NOT GASNet_ROOT_DIR)
    option(Legion_EMBED_GASNet "Embed a custom GASNet build into Legion" ON)
  endif()
endif()

set_or_default(Legion_MAX_DIM LEGION_MAX_DIM 4)

# Check the max dimensions
if((Legion_MAX_DIM LESS 1) OR (Legion_MAX_DIM GREATER 9))
  message(FATAL_ERROR "The maximum number of Legate dimensions must be between 1 and 9 inclusive")
endif()

set_or_default(Legion_MAX_FIELDS LEGION_MAX_FIELDS 256)

# Check that max fields is between 32 and 4096 and is a power of 2
if(NOT Legion_MAX_FIELDS MATCHES "^(32|64|128|256|512|1024|2048|4096)$")
  message(FATAL_ERROR "The maximum number of Legate fields must be a power of 2 between 32 and 4096 inclusive")
endif()

# We never want local fields
set(Legion_DEFAULT_LOCAL_FIELDS 0)

option(legate_core_STATIC_CUDA_RUNTIME "Statically link the cuda runtime library" OFF)
option(legate_core_EXCLUDE_LEGION_FROM_ALL "Exclude Legion targets from legate.core's 'all' target" OFF)
option(legate_core_COLLECTIVE "Use of collective instances" ON)
option(legate_core_BUILD_DOCS "Build doxygen docs" OFF)


set_or_default(NCCL_DIR NCCL_PATH)
set_or_default(Thrust_DIR THRUST_PATH)
set_or_default(CUDA_TOOLKIT_ROOT_DIR CUDA)
set_or_default(CMAKE_CUDA_ARCHITECTURES GPU_ARCH NATIVE)
set_or_default(Legion_HIJACK_CUDART USE_CUDART_HIJACK OFF)

include(CMakeDependentOption)
cmake_dependent_option(Legion_HIJACK_CUDART
  "Allow Legion to hijack and rewrite application calls into the CUDA runtime"
  ON
  "Legion_USE_CUDA;Legion_HIJACK_CUDART"
  OFF)
# This needs to be added as an option to force values to be visible in Legion build
option(Legion_HIJACK_CUDART "Replace default CUDA runtime with the Realm version" OFF)

if(Legion_HIJACK_CUDART)
  message(WARNING [=[
#####################################################################
Warning: Realm's CUDA runtime hijack is incompatible with NCCL.
Please note that your code will crash catastrophically as soon as it
calls into NCCL either directly or through some other Legate library.
#####################################################################
]=])
endif()

if(BUILD_SHARED_LIBS)
  if(Legion_HIJACK_CUDART)
    # Statically link CUDA if HIJACK_CUDART is set
    set(Legion_CUDA_DYNAMIC_LOAD OFF)
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  elseif(NOT DEFINED Legion_CUDA_DYNAMIC_LOAD)
    # If HIJACK_CUDART isn't set and BUILD_SHARED_LIBS is true, default Legion_CUDA_DYNAMIC_LOAD to true
    set(Legion_CUDA_DYNAMIC_LOAD ON)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  endif()
elseif(NOT DEFINED Legion_CUDA_DYNAMIC_LOAD)
  # If BUILD_SHARED_LIBS is false, default Legion_CUDA_DYNAMIC_LOAD to false also
  set(Legion_CUDA_DYNAMIC_LOAD OFF)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
endif()
