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

include_guard(GLOBAL)

function(find_or_configure_legion)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} Legion PKG_VERSION)

  string(REGEX REPLACE "^0([0-9]+)?$" "\\1" Legion_major_version "${Legion_major_version}")
  string(REGEX REPLACE "^0([0-9]+)?$" "\\1" Legion_minor_version "${Legion_minor_version}")
  string(REGEX REPLACE "^0([0-9]+)?$" "\\1" Legion_patch_version "${Legion_patch_version}")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Legion version git_repo git_branch shallow exclude_from_all)

  set(version "${Legion_major_version}.${Legion_minor_version}.${Legion_patch_version}")
  set(exclude_from_all ${PKG_EXCLUDE_FROM_ALL})
  if(PKG_BRANCH)
    set(git_branch "${PKG_BRANCH}")
  endif()
  if(PKG_REPOSITORY)
    set(git_repo "${PKG_REPOSITORY}")
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     Legion::Realm
                         Legion::Regent
                         Legion::Legion
                         Legion::RealmRuntime
                         Legion::LegionRuntime
      BUILD_EXPORT_SET   legate-core-exports
      INSTALL_EXPORT_SET legate-core-exports)

  if((NOT CPM_Legion_SOURCE) AND (NOT CPM_DOWNLOAD_Legion))
    # First try to find Legion via find_package()
    # so the `Legion_USE_*` variables are visible
    # Use QUIET find by default.
    set(_find_mode QUIET)
    # If Legion_DIR/Legion_ROOT are defined as something other than empty or NOTFOUND
    # use a REQUIRED find so that the build does not silently download Legion.
    if(Legion_DIR OR Legion_ROOT)
      set(_find_mode REQUIRED)
    endif()
    rapids_find_package(Legion ${version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})
  endif()

  if(Legion_FOUND)
    message(STATUS "CPM: using local package Legion@${version}")
  else()

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
    get_cpm_git_args(legion_cpm_git_args REPOSITORY ${git_repo} BRANCH ${git_branch})
    if(NOT DEFINED Legion_PYTHON_EXTRA_INSTALL_ARGS)
      set(Legion_PYTHON_EXTRA_INSTALL_ARGS "--root / --prefix \"\${CMAKE_INSTALL_PREFIX}\"")
    endif()

    # Support comma and semicolon delimited lists
    string(REPLACE "," " " Legion_PYTHON_EXTRA_INSTALL_ARGS "${Legion_PYTHON_EXTRA_INSTALL_ARGS}")
    string(REPLACE ";" " " Legion_PYTHON_EXTRA_INSTALL_ARGS "${Legion_PYTHON_EXTRA_INSTALL_ARGS}")

    set(_legion_cuda_options "")

    # Set CMAKE_CXX_STANDARD and CMAKE_CUDA_STANDARD for Legion builds. Legion's FindCUDA.cmake
    # use causes CUDA object compilation to fail if `-std=` flag is present in `CXXFLAGS` but
    # missing in `CUDA_NVCC_FLAGS`.
    set(_cxx_std "${CMAKE_CXX_STANDARD}")
    if(NOT _cxx_std)
      set(_cxx_std 17)
    endif()

    if(Legion_USE_CUDA)
      set(_cuda_std "${CMAKE_CUDA_STANDARD}")
      if(NOT _cuda_std)
        set(_cuda_std ${_cxx_std})
      endif()

      list(APPEND _legion_cuda_options "CMAKE_CUDA_STANDARD ${_cuda_std}")

      if(legate_core_STATIC_CUDA_RUNTIME)
        list(APPEND _legion_cuda_options "CMAKE_CUDA_RUNTIME_LIBRARY STATIC")
      else()
        list(APPEND _legion_cuda_options "CMAKE_CUDA_RUNTIME_LIBRARY SHARED")
      endif()
    endif()

    # Because legion sets these as cache variables, we need to force set this as a cache variable here
    # to ensure that Legion doesn't override this in the CMakeCache.txt and create an unexpected state.
    # This only applies to set() but does not apply to option() variables.
    # See discussion of FetchContent subtleties:
    # Only use these FORCE calls if using a Legion subbuild.
    # https://discourse.cmake.org/t/fetchcontent-cache-variables/1538/8
    set(Legion_MAX_DIM ${Legion_MAX_DIM} CACHE STRING "The max number of dimensions for Legion" FORCE)
    set(Legion_MAX_FIELDS ${Legion_MAX_FIELDS} CACHE STRING "The max number of fields for Legion" FORCE)
    set(Legion_DEFAULT_LOCAL_FIELDS ${Legion_DEFAULT_LOCAL_FIELDS} CACHE STRING "Number of local fields for Legion" FORCE)

    message(VERBOSE "legate.core: Legion version: ${version}")
    message(VERBOSE "legate.core: Legion git_repo: ${git_repo}")
    message(VERBOSE "legate.core: Legion git_branch: ${git_branch}")
    message(VERBOSE "legate.core: Legion exclude_from_all: ${exclude_from_all}")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(Legion_BACKTRACE_USE_LIBDW ON)
    else()
      set(Legion_BACKTRACE_USE_LIBDW OFF)
    endif()

    rapids_cpm_find(Legion ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legion_cpm_git_args}
          FIND_PACKAGE_ARGUMENTS EXACT
          EXCLUDE_FROM_ALL       ${exclude_from_all}
          OPTIONS                ${_legion_cuda_options}
                                 "CMAKE_CXX_STANDARD ${_cxx_std}"
                                 "Legion_VERSION ${version}"
                                 "Legion_BUILD_BINDINGS ON"
                                 "Legion_REDOP_HALF ON"
                                 "Legion_REDOP_COMPLEX ON"
                                 "Legion_BUILD_RUST_PROFILER ON"
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_USE_Python ${Legion_USE_Python} PARENT_SCOPE)
  set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)
  set(Legion_NETWORKS ${Legion_NETWORKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_USE_Python=${Legion_USE_Python}")
  message(VERBOSE "Legion_CUDA_ARCH=${Legion_CUDA_ARCH}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
  message(VERBOSE "Legion_NETWORKS=${Legion_NETWORKS}")

endfunction()

foreach(_var IN ITEMS "legate_core_LEGION_VERSION"
                      "legate_core_LEGION_BRANCH"
                      "legate_core_LEGION_REPOSITORY"
                      "legate_core_EXCLUDE_LEGION_FROM_ALL")
  if(DEFINED ${_var})
    # Create a legate_core_LEGION_BRANCH variable in the current scope either from the existing
    # current-scope variable, or the cache variable.
    set(${_var} "${${_var}}")
    # Remove legate_core_LEGION_BRANCH from the CMakeCache.txt. This ensures reconfiguring the same
    # build dir without passing `-Dlegate_core_LEGION_BRANCH=` reverts to the value in versions.json
    # instead of reusing the previous `-Dlegate_core_LEGION_BRANCH=` value.
    unset(${_var} CACHE)
  endif()
endforeach()

if(NOT DEFINED legate_core_LEGION_VERSION)
  set(legate_core_LEGION_VERSION "${legate_core_VERSION}")
endif()

find_or_configure_legion(VERSION          ${legate_core_LEGION_VERSION}
                         REPOSITORY       ${legate_core_LEGION_REPOSITORY}
                         BRANCH           ${legate_core_LEGION_BRANCH}
                         EXCLUDE_FROM_ALL ${legate_core_EXCLUDE_LEGION_FROM_ALL}
)
