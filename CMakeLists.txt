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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.06/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build legate.core shared libraries" ON)

function(set_or_default var_name var_env)
  list(LENGTH ARGN num_extra_args)
  if(num_extra_args GREATER 0)
    list(GET ARGN 0 var_default)
  endif()
  if(DEFINED ${var_name})
    set(${var_name} ${${var_name}} PARENT_SCOPE)
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
set_or_default(Legion_USE_GASNet USE_GASNET OFF)
set_or_default(Legion_USE_OpenMP USE_OPENMP OFF)
set_or_default(Legion_BOUNDS_CHECKS CHECK_BOUNDS OFF)
set_or_default(Legion_MAX_DIM LEGION_MAX_DIM 4)

# Check the max dimensions
if ((Legion_MAX_DIM LESS 1) OR (Legion_MAX_DIM GREATER 9))
  message(FATAL_ERROR "The maximum number of Legate dimensions must be between 1 and 9 inclusive")
endif()

set_or_default(Legion_MAX_FIELDS LEGION_MAX_FIELDS 256)

# Check that max fields is between 32 and 4096 and is a power of 2
if (NOT Legion_MAX_FIELDS MATCHES "^(32|64|128|256|512|1024|2048|4096)$")
  message(FATAL_ERROR "The maximum number of Legate fields must be a power of 2 between 32 and 4096 inclusive")
endif()

set_or_default(GASNet_ROOT_DIR GASNET)
set_or_default(NCCL_DIR NCCL_PATH)
set_or_default(Thrust_DIR THRUST_PATH)
set_or_default(CUDA_TOOLKIT_ROOT_DIR CUDA)

option(Legion_SPY "Enable detailed logging for Legion Spy" OFF)
option(Legion_USE_LLVM "Use LLVM JIT operations" OFF)
option(Legion_USE_CUDA "Enable Legion support for the CUDA runtime" OFF)
option(Legion_USE_HDF5 "Enable support for HDF5" OFF)
option(Legion_USE_GASNet "Enable the distributed GASNet backend" OFF)
option(Legion_USE_OpenMP "Use OpenMP" OFF)
option(Legion_USE_Python "Use Python" OFF)

if(Legion_USE_LLVM)
  # Prevent `foreach` variables from leaking. FindLLVM.cmake does:
  # ```
  # foreach(C IN LISTS LLVM_FIND_COMPONENTS)
  # ```
  # This changes the value of the `C` variable globally, interfering with FindMPI.cmake
  cmake_policy(SET CMP0124 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0124 NEW)
endif()

if(Legion_USE_GASNet AND (NOT GASNet_ROOT_DIR))
  option(Legion_EMBED_GASNet "Embed a custom GASNet build into Legion" ON)
endif()

set_or_default(GASNet_CONDUIT CONDUIT "udp")

option(LEGATE_CORE_EXCLUDE_LEGION_FROM_ALL "Exclude Legion targets from legate.core's 'all' target" OFF)
set(LEGATE_CORE_LEGION_REPOSITORY https://gitlab.com/StanfordLegion/legion.git)
set(LEGATE_CORE_LEGION_BRANCH control_replication)

include(CMakeDependentOption)
cmake_dependent_option(Legion_HIJACK_CUDART
  "Allow Legion to hijack and rewrite application calls into the CUDA runtime"
  ON
  Legion_USE_CUDA;"DEFINED ENV{USE_CUDART_HIJACK} AND $ENV{USE_CUDART_HIJACK} STREQUAL 1"
  OFF)

if(Legion_USE_CUDA)
  if(BUILD_SHARED_LIBS)
    if(Legion_HIJACK_CUDART)
      # Must statically link CUDA if HIJACK_CUDART is set
      set(Legion_CUDA_DYNAMIC_LOAD OFF)
    elseif(NOT DEFINED Legion_CUDA_DYNAMIC_LOAD)
      # If HIJACK_CUDART isn't set and BUILD_SHARED_LIBS is true, default Legion_CUDA_DYNAMIC_LOAD to true
      set(Legion_CUDA_DYNAMIC_LOAD ON)
    endif()
  elseif(NOT DEFINED Legion_CUDA_DYNAMIC_LOAD)
    # If BUILD_SHARED_LIBS is false, default Legion_CUDA_DYNAMIC_LOAD to false also
    set(Legion_CUDA_DYNAMIC_LOAD OFF)
  endif()
endif()

if(Legion_USE_CUDA)
  # Needs to run before `project` call
  rapids_cuda_init_architectures(LEGATE_CORE)
  project(LEGATE_CORE VERSION 22.05.00 LANGUAGES C CXX CUDA)
else()
  project(LEGATE_CORE VERSION 22.05.00 LANGUAGES C CXX)
endif()

# Write the version header
rapids_cmake_write_version_file(include/legate/version_config.hpp)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(Legion_USE_CUDA)
  include(cmake/thirdparty/get_nccl.cmake)
  include(cmake/thirdparty/get_thrust.cmake)
endif()

if(Legion_USE_Python)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "legate.core: Has Python3: ${Python3_FOUND}")
  message(STATUS "legate.core: Has Python 3 interpreter: ${Python3_Interpreter_FOUND}")
  message(STATUS "legate.core: Python 3 include directories: ${Python3_INCLUDE_DIRS}")
  message(STATUS "legate.core: Python 3 libraries: ${Python3_LIBRARIES}")
  message(STATUS "legate.core: Python 3 library directories: ${Python3_LIBRARY_DIRS}")
  message(STATUS "legate.core: Python 3 version: ${Python3_VERSION}")
  set(Legion_Python_Version ${Python3_VERSION})
endif()

include(cmake/thirdparty/get_legion.cmake)

##############################################################################
# - legate.core --------------------------------------------------------------

set(LEGATE_CORE_SOURCES "")
set(LEGATE_CORE_CXX_DEFS "")
set(LEGATE_CORE_CUDA_DEFS "")
set(LEGATE_CORE_CXX_OPTIONS "")
set(LEGATE_CORE_CUDA_OPTIONS "")

if(Legion_USE_CUDA)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_CUDA)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_CUDA)
endif()

if(Legion_USE_OpenMP)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_OPENMP)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_OPENMP)
endif()

if(Legion_USE_GASNet)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_GASNET)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_GASNET)
endif()

list(APPEND LEGATE_CORE_SOURCES
  src/core/legate_c.cc
  src/core/comm/comm.cc
  src/core/data/allocator.cc
  src/core/data/scalar.cc
  src/core/data/store.cc
  src/core/data/transform.cc
  src/core/mapping/base_mapper.cc
  src/core/mapping/core_mapper.cc
  src/core/mapping/instance_manager.cc
  src/core/mapping/mapping.cc
  src/core/mapping/task.cc
  src/core/runtime/context.cc
  src/core/runtime/projection.cc
  src/core/runtime/runtime.cc
  src/core/runtime/shard.cc
  src/core/task/return.cc
  src/core/task/task.cc
  src/core/utilities/debug.cc
  src/core/utilities/deserializer.cc
  src/core/utilities/machine.cc
  src/core/utilities/linearize.cc
)

if(Legion_USE_CUDA)
  list(APPEND LEGATE_CORE_SOURCES src/core/comm/comm_nccl.cu)
endif()

add_library(legate_core ${LEGATE_CORE_SOURCES})
add_library(legate::core ALIAS legate_core)

set_target_properties(legate_core
           PROPERTIES EXPORT_NAME                         core
                      LIBRARY_OUTPUT_NAME                 lgcore
                      BUILD_RPATH                         "\$ORIGIN"
                      INSTALL_RPATH                       "\$ORIGIN"
                      CXX_STANDARD                        14
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_link_libraries(legate_core PUBLIC Legion::Legion)
target_compile_options(legate_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LEGATE_CORE_CXX_OPTIONS}>")
target_compile_definitions(legate_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LEGATE_CORE_CXX_DEFS}>")
target_include_directories(legate_core
  PRIVATE
    $<BUILD_INTERFACE:${LEGATE_CORE_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include/legate>
)

if(Legion_USE_CUDA)
  set_target_properties(legate_core
             PROPERTIES CUDA_STANDARD                     14
                        CUDA_STANDARD_REQUIRED            ON)

  list(APPEND LEGATE_CORE_CUDA_OPTIONS --expt-relaxed-constexpr)
  list(APPEND LEGATE_CORE_CUDA_OPTIONS --expt-extended-lambda)
  list(APPEND LEGATE_CORE_CUDA_OPTIONS -Xfatbin=-compress-all)

  target_link_libraries(legate_core PUBLIC legate::Thrust NCCL::NCCL)
  target_compile_options(legate_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${LEGATE_CORE_CUDA_OPTIONS}>")
  target_compile_definitions(legate_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${LEGATE_CORE_CUDA_DEFS}>")

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
          [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(legate_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_core
        DESTINATION ${lib_dir}
        EXPORT legate-core-exports)

install(
  FILES src/legate.h
        src/legate_defines.h
        src/legate_preamble.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/legate/version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate)

install(
  FILES src/core/legate_c.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core)

install(
  FILES src/core/comm/communicator.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/comm)

install(
  FILES src/core/data/allocator.h
        src/core/data/buffer.h
        src/core/data/scalar.h
        src/core/data/scalar.inl
        src/core/data/store.h
        src/core/data/store.inl
        src/core/data/transform.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/data)

install(
  FILES src/core/mapping/base_mapper.h
        src/core/mapping/mapping.h
        src/core/mapping/task.h
        src/core/mapping/task.inl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/mapping)

install(
  FILES src/core/runtime/context.h
        src/core/runtime/runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/runtime)

install(
  FILES src/core/task/return.h
        src/core/task/task.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/task)

install(
  FILES src/core/utilities/debug.h
        src/core/utilities/deserializer.h
        src/core/utilities/deserializer.inl
        src/core/utilities/dispatch.h
        src/core/utilities/machine.h
        src/core/utilities/span.h
        src/core/utilities/type_traits.h
        src/core/utilities/typedefs.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/utilities)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for Legate Core, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::core

]=])

set(code_string "")

if(Legion_USE_CUDA)
  set(code_string
[=[
if(NOT TARGET legate::Thrust)
  thrust_create_target(legate::Thrust FROM_OPTIONS)
endif()
]=])
endif()

rapids_export(
  INSTALL legate_core
  EXPORT_SET legate-core-exports
  GLOBAL_TARGETS core
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_core
  EXPORT_SET legate-core-exports
  GLOBAL_TARGETS core
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
