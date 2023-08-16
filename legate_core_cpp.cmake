#=============================================================================
# Copyright 2022-2023 NVIDIA Corporation
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

##############################################################################
# - User Options  ------------------------------------------------------------

option(legate_core_BUILD_TESTS OFF)
option(legate_core_BUILD_EXAMPLES OFF)
include(cmake/Modules/legate_core_options.cmake)

##############################################################################
# - Project definition -------------------------------------------------------

# Write the version header
rapids_cmake_write_version_file(include/legate/version_config.hpp)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

##############################################################################
# - conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
# Do this before we include Legion, so its build also inherits this setting.
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

macro(_find_package_Python3)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  message(VERBOSE "legate.core: Has Python3: ${Python3_FOUND}")
  message(VERBOSE "legate.core: Has Python 3 interpreter: ${Python3_Interpreter_FOUND}")
  message(VERBOSE "legate.core: Python 3 include directories: ${Python3_INCLUDE_DIRS}")
  message(VERBOSE "legate.core: Python 3 libraries: ${Python3_LIBRARIES}")
  message(VERBOSE "legate.core: Python 3 library directories: ${Python3_LIBRARY_DIRS}")
  message(VERBOSE "legate.core: Python 3 version: ${Python3_VERSION}")
endmacro()

if(Legion_USE_Python)
  _find_package_Python3()
  if(Python3_FOUND AND Python3_VERSION)
    set(Legion_Python_Version ${Python3_VERSION})
  endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cuda_arch_helpers.cmake)

if(Legion_USE_CUDA)
  # Needs to run before find_package(Legion)
  set_cuda_arch_from_names()
endif()

###
# If we find Legion already configured on the system, it will report whether it
# was compiled with Python (Legion_USE_PYTHON), CUDA (Legion_USE_CUDA), OpenMP
# (Legion_USE_OpenMP), and networking (Legion_NETWORKS).
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

if(Legion_NETWORKS)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(Legion_USE_CUDA)
  # Enable the CUDA language
  enable_language(CUDA)
  # Must come after `enable_language(CUDA)`
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")
  # Find the CUDAToolkit
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET legate-core-exports
    INSTALL_EXPORT_SET legate-core-exports
  )
  # Find NCCL
  include(cmake/thirdparty/get_nccl.cmake)
endif()

# Find or install Thrust
include(cmake/thirdparty/get_thrust.cmake)

##############################################################################
# - legate.core --------------------------------------------------------------

set(legate_core_SOURCES "")
set(legate_core_CXX_DEFS "")
set(legate_core_CUDA_DEFS "")
set(legate_core_CXX_OPTIONS "")
set(legate_core_CUDA_OPTIONS "")

include(cmake/Modules/set_cpu_arch_flags.cmake)
set_cpu_arch_flags(legate_core_CXX_OPTIONS)

if (legate_core_COLLECTIVE)
  list(APPEND legate_core_CXX_DEFS LEGATE_USE_COLLECTIVE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND legate_core_CXX_DEFS DEBUG_LEGATE)
  list(APPEND legate_core_CUDA_DEFS DEBUG_LEGATE)
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_core_CXX_DEFS LEGATE_USE_CUDA)
  list(APPEND legate_core_CUDA_DEFS LEGATE_USE_CUDA)

  add_cuda_architecture_defines(legate_core_CUDA_DEFS ARCHS ${Legion_CUDA_ARCH})

  list(APPEND legate_core_CUDA_OPTIONS -Xfatbin=-compress-all)
  list(APPEND legate_core_CUDA_OPTIONS --expt-extended-lambda)
  list(APPEND legate_core_CUDA_OPTIONS --expt-relaxed-constexpr)
  list(APPEND legate_core_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
endif()

if(Legion_USE_OpenMP)
  list(APPEND legate_core_CXX_DEFS LEGATE_USE_OPENMP)
  list(APPEND legate_core_CUDA_DEFS LEGATE_USE_OPENMP)
endif()

if(Legion_NETWORKS)
  list(APPEND legate_core_CXX_DEFS LEGATE_USE_NETWORK)
  list(APPEND legate_core_CUDA_DEFS LEGATE_USE_NETWORK)
endif()

# Change THRUST_DEVICE_SYSTEM for `.cpp` files
# TODO: This is what we do in cuNumeric, should we do it here as well?
if(Legion_USE_OpenMP)
  list(APPEND legate_core_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_core_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(NOT Legion_USE_CUDA)
  list(APPEND legate_core_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_core_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()
# Or should we only do it if OpenMP and CUDA are both disabled?
# if(NOT Legion_USE_OpenMP AND (NOT Legion_USE_CUDA))
#   list(APPEND legate_core_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
#   list(APPEND legate_core_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
# endif()

list(APPEND legate_core_SOURCES
  src/core/legate_c.cc
  src/core/comm/comm.cc
  src/core/comm/comm_cpu.cc
  src/core/comm/coll.cc
  src/core/data/allocator.cc
  src/core/data/scalar.cc
  src/core/data/store.cc
  src/core/data/transform.cc
  src/core/mapping/base_mapper.cc
  src/core/mapping/core_mapper.cc
  src/core/mapping/default_mapper.cc
  src/core/mapping/instance_manager.cc
  src/core/mapping/machine.cc
  src/core/mapping/mapping.cc
  src/core/mapping/operation.cc
  src/core/mapping/store.cc
  src/core/runtime/context.cc
  src/core/runtime/projection.cc
  src/core/runtime/runtime.cc
  src/core/runtime/shard.cc
  src/core/task/registrar.cc
  src/core/task/return.cc
  src/core/task/task.cc
  src/core/task/task_info.cc
  src/core/task/variant_options.cc
  src/core/type/type_info.cc
  src/core/utilities/debug.cc
  src/core/utilities/deserializer.cc
  src/core/utilities/machine.cc
  src/core/utilities/linearize.cc
)

if(Legion_NETWORKS)
  list(APPEND legate_core_SOURCES
    src/core/comm/mpi_comm.cc
    src/core/comm/local_comm.cc)
else()
  list(APPEND legate_core_SOURCES
    src/core/comm/local_comm.cc)
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_core_SOURCES
    src/core/comm/comm_nccl.cu
    src/core/cuda/stream_pool.cu)
endif()

add_library(legate_core ${legate_core_SOURCES})
add_library(legate::core ALIAS legate_core)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
endif ()

set_target_properties(legate_core
           PROPERTIES EXPORT_NAME                         core
                      LIBRARY_OUTPUT_NAME                 lgcore
                      BUILD_RPATH                         "${platform_rpath_origin}"
                      INSTALL_RPATH                       "${platform_rpath_origin}"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

if(Legion_USE_CUDA)
  set_property(TARGET legate_core PROPERTY CUDA_ARCHITECTURES ${Legion_CUDA_ARCH})
endif()

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(legate_core PRIVATE conda_env)
endif()

if(Legion_USE_CUDA)
  if(legate_core_STATIC_CUDA_RUNTIME)
    set_target_properties(legate_core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate_core PUBLIC CUDA::cudart_static)
  else()
    set_target_properties(legate_core PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(legate_core PUBLIC CUDA::cudart)
  endif()
endif()

target_link_libraries(legate_core
   PUBLIC Legion::Legion
          legate::Thrust
          $<TARGET_NAME_IF_EXISTS:CUDA::nvToolsExt>
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
  PRIVATE $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>)

target_compile_options(legate_core
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${legate_core_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${legate_core_CUDA_OPTIONS}>")

target_compile_definitions(legate_core
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${legate_core_CXX_DEFS}>"
         "$<$<COMPILE_LANGUAGE:CUDA>:${legate_core_CUDA_DEFS}>")

target_include_directories(legate_core
  PUBLIC
    $<BUILD_INTERFACE:${legate_core_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include/legate>
)

##############################################################################
# - Doxygen target------------------------------------------------------------

if (legate_core_BUILD_DOCS)
  find_package(Doxygen)
  if(Doxygen_FOUND)
    set(legate_core_DOC_SOURCES "")
    list(APPEND legate_core_DOC_SOURCES
      # type
      src/core/type/type_info.h
      src/core/type/type_traits.h
      # task
      src/core/task/task.h
      src/core/task/registrar.h
      src/core/task/variant_options.h
      src/core/task/exception.h
      src/core/cuda/stream_pool.h
      # data
      src/core/data/store.h
      src/core/data/scalar.h
      src/core/data/buffer.h
      src/core/utilities/span.h
      src/core/data/allocator.h
      # runtime
      src/core/runtime/runtime.h
      src/core/runtime/runtime.inl
      src/core/runtime/context.h
      # mapping
      src/core/mapping/mapping.h
      src/core/mapping/operation.h
      # aliases
      src/core/utilities/typedefs.h
      # utilities
      src/core/utilities/debug.h
      src/core/utilities/dispatch.h
      # main page
      src/legate.h
    )
    set(DOXYGEN_PROJECT_NAME "Legate")
    set(DOXYGEN_FULL_PATH_NAMES NO)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTENSION_MAPPING cu=C++ cuh=C++)
    set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
    set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
    set(DOXYGEN_STRIP_FROM_INC_PATH ${CMAKE_SOURCE_DIR}/src)
    doxygen_add_docs("doxygen_legate" ALL
      ${legate_core_DOC_SOURCES}
      COMMENT "Custom command for building Doxygen docs."
    )
  else()
    message(STATUS "cannot find Doxygen. not generating docs.")
  endif()
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
  FILES src/core/comm/coll.h
        src/core/comm/communicator.h
        src/core/comm/pthread_barrier.h
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
  FILES src/core/mapping/machine.h
        src/core/mapping/mapping.h
        src/core/mapping/operation.h
        src/core/mapping/operation.inl
        src/core/mapping/store.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/mapping)

install(
  FILES src/core/runtime/context.h
        src/core/runtime/context.inl
        src/core/runtime/resource.h
        src/core/runtime/runtime.h
        src/core/runtime/runtime.inl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/runtime)

install(
  FILES src/core/task/exception.h
        src/core/task/registrar.h
        src/core/task/return.h
        src/core/task/task.h
        src/core/task/task.inl
        src/core/task/task_info.h
        src/core/task/variant_helper.h
        src/core/task/variant_options.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/task)

install(
  FILES src/core/type/type_info.h
        src/core/type/type_traits.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate/core/type)
install(
  FILES src/core/utilities/debug.h
        src/core/utilities/deserializer.h
        src/core/utilities/deserializer.inl
        src/core/utilities/dispatch.h
        src/core/utilities/machine.h
        src/core/utilities/nvtx_help.h
        src/core/utilities/span.h
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

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/cmake/legate_helper_functions.cmake helper_functions)

string(JOIN "\n" code_string
[=[
if(NOT TARGET legate::Thrust)
  thrust_create_target(legate::Thrust FROM_OPTIONS)
endif()
]=]
  "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
  "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
  "set(Legion_USE_Python ${Legion_USE_Python})"
  "set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH})"
  "set(Legion_NETWORKS ${Legion_NETWORKS})"
  "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
[=[
if(Legion_NETWORKS)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()
]=]
"${helper_functions}"
)

rapids_export(
  INSTALL legate_core
  EXPORT_SET legate-core-exports
  GLOBAL_TARGETS core
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD legate_core
  EXPORT_SET legate-core-exports
  GLOBAL_TARGETS core
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
  LANGUAGES ${ENABLED_LANGUAES}
)

include(cmake/legate_helper_functions.cmake)

set(legate_core_ROOT ${CMAKE_CURRENT_BINARY_DIR})

if(legate_core_BUILD_TESTS)
  add_subdirectory(tests/integration)
endif()

if(legate_core_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
