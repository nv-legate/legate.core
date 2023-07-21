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

##############################################################################
# - User Options  ------------------------------------------------------------

option(FIND_LEGATE_CORE_CPP "Search for existing legate_core C++ installations before defaulting to local files"
       OFF)

##############################################################################
# - Dependencies -------------------------------------------------------------

# If the user requested it we attempt to find legate_core.
if(FIND_LEGATE_CORE_CPP)
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${legate_core_version} legate_core parsed_ver)
  rapids_find_package(legate_core ${parsed_ver} EXACT CONFIG
                      GLOBAL_TARGETS     legate::core
                      BUILD_EXPORT_SET   legate-core-python-exports
                      INSTALL_EXPORT_SET legate-core-python-exports)
else()
  set(legate_core_FOUND OFF)
endif()

if(NOT legate_core_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  set(Legion_BUILD_BINDINGS ON)
  add_subdirectory(. "${CMAKE_CURRENT_SOURCE_DIR}/build")
  get_target_property(cython_lib_dir legate_core LIBRARY_OUTPUT_DIRECTORY)
  set(cython_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/build/${cython_lib_dir}")
  set(SKBUILD ON)
endif()

add_custom_target("generate_install_info_py" ALL
  COMMAND ${CMAKE_COMMAND}
          -DLegion_NETWORKS="${Legion_NETWORKS}"
          -DGASNet_CONDUIT="${GASNet_CONDUIT}"
          -DLegion_USE_CUDA="${Legion_USE_CUDA}"
          -DLegion_USE_OpenMP="${Legion_USE_OpenMP}"
          -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_install_info_py.cmake"
  COMMENT "Generate install_info.py"
)

add_library(legate_core_python INTERFACE)
add_library(legate::core_python ALIAS legate_core_python)
target_link_libraries(legate_core_python INTERFACE legate::core)

include(rapids-cython)
rapids_cython_init()

add_subdirectory(legate/core/_lib)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET legate_core PATHS "${cython_lib_dir}")
endif()

##############################################################################
# - conda environment --------------------------------------------------------

rapids_cmake_support_conda_env(conda_env)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_core_python
        DESTINATION ${lib_dir}
        EXPORT legate-core-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for Legate Python, the Foundation for All Legate Libraries.

Imported Targets:
  - legate::core_python

]=])

set(code_string "")

rapids_export(
  INSTALL legate_core_python
  EXPORT_SET legate-core-python-exports
  GLOBAL_TARGETS core_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_core_python
  EXPORT_SET legate-core-python-exports
  GLOBAL_TARGETS core_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
