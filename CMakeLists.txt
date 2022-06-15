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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

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

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
  cmake_policy(SET CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
  cmake_policy(SET CMP0126 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
endif()

##############################################################################
# - Project definition -------------------------------------------------------

project(LEGATE_CORE VERSION 22.07.00 LANGUAGES C CXX)

# Write the version header
rapids_cmake_write_version_file(include/legate/version_config.hpp)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

include(cmake/Modules/legate_core_options.cmake)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

function(_find_package_Python3)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  message(VERBOSE "legate.core: Has Python3: ${Python3_FOUND}")
  message(VERBOSE "legate.core: Has Python 3 interpreter: ${Python3_Interpreter_FOUND}")
  message(VERBOSE "legate.core: Python 3 include directories: ${Python3_INCLUDE_DIRS}")
  message(VERBOSE "legate.core: Python 3 libraries: ${Python3_LIBRARIES}")
  message(VERBOSE "legate.core: Python 3 library directories: ${Python3_LIBRARY_DIRS}")
  message(VERBOSE "legate.core: Python 3 version: ${Python3_VERSION}")
endfunction()

if(Legion_USE_Python)
  _find_package_Python3()
  if(Python3_FOUND AND Python3_VERSION)
    set(Legion_Python_Version ${Python3_VERSION})
  endif()
endif()

###
# If we find Legion already configured on the system, it will report whether it
# was compiled with Python (Legion_USE_PYTHON), CUDA (Legion_USE_CUDA), OpenMP
# (Legion_USE_OpenMP), and GASNet (Legion_USE_GASNet).
#
# We use the same variables as Legion because we want to enable/disable each of
# these features based on how Legion was configured (it doesn't make sense to
# build legate.core's Python bindings if Legion's bindings weren't compiled).
###
include(cmake/thirdparty/get_legion.cmake)

# If Legion_USE_Python was toggled ON by find_package(Legion), find Python3
if(Legion_USE_Python AND (NOT Python3_FOUND))
  _find_package_Python3()
endif()

if(Legion_USE_CUDA)
  # Find the CUDAToolkit
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET legate-core-exports
    INSTALL_EXPORT_SET legate-core-exports
  )

  # Find NCCL
  include(cmake/thirdparty/get_nccl.cmake)

  # Enable the CUDA language
  enable_language(CUDA)

  # Must come after `enable_language(CUDA)`
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")
endif()

# Find or install Thrust
include(cmake/thirdparty/get_thrust.cmake)

##############################################################################
# - legate.core --------------------------------------------------------------

set(LEGATE_CORE_SOURCES "")
set(LEGATE_CORE_CXX_DEFS "")
set(LEGATE_CORE_CUDA_DEFS "")
set(LEGATE_CORE_CXX_OPTIONS "")
set(LEGATE_CORE_CUDA_OPTIONS "")

include(cmake/Modules/set_cpu_arch_flags.cmake)
set_cpu_arch_flags(LEGATE_CORE_CXX_OPTIONS)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND LEGATE_CORE_CXX_DEFS DEBUG_LEGATE)
  list(APPEND LEGATE_CORE_CUDA_DEFS DEBUG_LEGATE)
endif()

if(Legion_USE_CUDA)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_CUDA)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_CUDA)

  add_cuda_architecture_defines(LEGATE_CORE_CUDA_DEFS)

  list(APPEND LEGATE_CORE_CUDA_OPTIONS -Xfatbin=-compress-all)
  list(APPEND LEGATE_CORE_CUDA_OPTIONS --expt-extended-lambda)
  list(APPEND LEGATE_CORE_CUDA_OPTIONS --expt-relaxed-constexpr)
endif()

if(Legion_USE_OpenMP)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_OPENMP)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_OPENMP)
endif()

if(Legion_USE_GASNet)
  list(APPEND LEGATE_CORE_CXX_DEFS LEGATE_USE_GASNET)
  list(APPEND LEGATE_CORE_CUDA_DEFS LEGATE_USE_GASNET)
endif()

# Change THRUST_DEVICE_SYSTEM for `.cpp` files
# TODO: This is what we do in cuNumeric, should we do it here as well?
if(Legion_USE_OpenMP)
  list(APPEND LEGATE_CORE_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND LEGATE_CORE_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(NOT Legion_USE_CUDA)
  list(APPEND LEGATE_CORE_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND LEGATE_CORE_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()
# Or should we only do it if OpenMP and CUDA are both disabled?
# if(NOT Legion_USE_OpenMP AND (NOT Legion_USE_CUDA))
#   list(APPEND LEGATE_CORE_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
#   list(APPEND LEGATE_CORE_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
# endif()

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
  list(APPEND LEGATE_CORE_SOURCES
  src/core/comm/comm_nccl.cu
  src/core/cuda/stream_pool.cu)
endif()

add_library(legate_core ${LEGATE_CORE_SOURCES})
add_library(legate::core ALIAS legate_core)

set_target_properties(legate_core
           PROPERTIES EXPORT_NAME                         core
                      LIBRARY_OUTPUT_NAME                 lgcore
                      BUILD_RPATH                         "\$ORIGIN"
                      INSTALL_RPATH                       "\$ORIGIN"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(legate_core PRIVATE conda_env)
endif()

if(Legion_USE_CUDA)
  if(LEGATE_CORE_STATIC_CUDA_RUNTIME)
    set_target_properties(legate_core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate_core PUBLIC CUDA::cudart_static)
  else()
    set_target_properties(legate_core PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate_core PUBLIC CUDA::cudart)
  endif()
endif()

###
# Preinclude `realm_defines.h` and `legion_defines.h` in all compilations.
# TODO: It isn't possible to expose this as part of the public legate_core
# target's interface in a portable way, so this code needs to be duplicated
# in both legate.core and cuNumeric.
#
# A better solution would be to add #include directives to the requisite
# legate.core and cuNumeric headers.
###
include(cmake/Modules/legion_helpers.cmake)
get_legion_and_realm_includes(extra_include_options)

target_link_libraries(legate_core
   PUBLIC Legion::Legion
          legate::Thrust
          $<TARGET_NAME_IF_EXISTS:CUDA::nvToolsExt>
  PRIVATE $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>)

target_compile_options(legate_core
  PRIVATE ${extra_include_options}
          "$<$<COMPILE_LANGUAGE:CXX>:${LEGATE_CORE_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${LEGATE_CORE_CUDA_OPTIONS}>")

target_compile_definitions(legate_core
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${LEGATE_CORE_CXX_DEFS}>"
         "$<$<COMPILE_LANGUAGE:CUDA>:${LEGATE_CORE_CUDA_DEFS}>")

target_include_directories(legate_core
  PUBLIC
    $<BUILD_INTERFACE:${LEGATE_CORE_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include/legate>
)

if(Legion_USE_CUDA)
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

add_custom_target(legate_c_h_c ALL
  COMMENT "Preprocess src/core/legate_c.h for Python CFFI bindings"
  COMMAND ${CMAKE_C_COMPILER} -E -DLEGATE_USE_PYTHON_CFFI
          "-I$<JOIN:$<TARGET_PROPERTY:Legion::Legion,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/src/core/legate_c.h"
          -o "${CMAKE_CURRENT_BINARY_DIR}/include/legate/legate_c.h.i"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/include/legate/legate_c.h.i"
  COMMAND_EXPAND_LISTS
)
# Generate a target that depends on legate_c_h_c
add_library(legate_c_h_i INTERFACE)
add_dependencies(legate_c_h_i legate_c_h_c)

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
        ${CMAKE_CURRENT_BINARY_DIR}/include/legate/legate_c.h.i
        ${CMAKE_CURRENT_BINARY_DIR}/include/legate/version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate)

install(
  FILES src/core/legate_c.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core)

install(
  FILES src/core/comm/communicator.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/comm)

install(
  FILES src/core/cuda/cuda_help.h
        src/core/cuda/stream_pool.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/cuda)

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
  FILES src/core/task/exception.h
        src/core/task/return.h
        src/core/task/task.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/task)

install(
  FILES src/core/utilities/debug.h
        src/core/utilities/deserializer.h
        src/core/utilities/deserializer.inl
        src/core/utilities/dispatch.h
        src/core/utilities/machine.h
        src/core/utilities/nvtx_help.h
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

string(JOIN "\n" code_string
[=[
if(NOT TARGET legate::Thrust)
  thrust_create_target(legate::Thrust FROM_OPTIONS)
endif()
]=]
  "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
  "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
  "set(Legion_USE_Python ${Legion_USE_Python})"
  "set(Legion_USE_GASNet ${Legion_USE_GASNet})"
  "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
)

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
