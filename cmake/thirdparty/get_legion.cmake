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

function(find_or_configure_legion)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} Legion PKG_VERSION)

  set(Legion_CUDA_ARCH "")
  if(Legion_USE_CUDA)
    set(Legion_CUDA_ARCH ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-real" "")
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-virtual" "")
    list(JOIN Legion_CUDA_ARCH "," Legion_CUDA_ARCH)
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     Legion::Realm
                         Legion::Regent
                         Legion::Legion
                         Legion::RealmRuntime
                         Legion::LegionRuntime
      BUILD_EXPORT_SET   legate-core-exports
      INSTALL_EXPORT_SET legate-core-exports)

  # First try to find Legion via find_package()
  # so the `Legion_USE_*` variables are visible
  # Use QUIET find by default.
  set(_find_mode QUIET)
  # If Legion_DIR/Legion_ROOT are defined as something other than empty or NOTFOUND
  # use a REQUIRED find so that the build does not silently download Legion.
  if(Legion_DIR OR Legion_ROOT)
    set(_find_mode REQUIRED)
  endif()
  rapids_find_package(Legion ${PKG_VERSION} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})

  if(Legion_FOUND)
    message(STATUS "CPM: using local package Legion@${PKG_VERSION}")
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
    get_cpm_git_args(legion_cpm_git_args REPOSITORY ${PKG_REPOSITORY} BRANCH ${PKG_BRANCH})
    if(NOT DEFINED Legion_PYTHON_EXTRA_INSTALL_ARGS)
      set(Legion_PYTHON_EXTRA_INSTALL_ARGS "--single-version-externally-managed --root=/")
    endif()

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

      if(NOT CUDA_NVCC_FLAGS)
        list(APPEND CUDA_NVCC_FLAGS "${CUDAFLAGS}")
      endif()

      set(_nvcc_flags ${CUDA_NVCC_FLAGS})
      if(NOT "${_nvcc_flags}" MATCHES "-std=")
        list(APPEND _nvcc_flags "-std=c++${_cuda_std}")
      endif()

      # Detect the presence of LIBRARY_PATH envvar so we can set
      # `CMAKE_LIBRARY_PATH` for Legion's FindCUDA.cmake calls.
      set(_lib_path "${CMAKE_LIBRARY_PATH}")
      if(DEFINED ENV{LIBRARY_PATH})
        list(APPEND _lib_path "$ENV{LIBRARY_PATH}")
      endif()

      if(NOT "${_lib_path}" MATCHES "stubs")
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
          list(APPEND _lib_path "${CUDAToolkit_LIBRARY_ROOT}/lib64/stubs")
        else()
          list(APPEND _lib_path "${CUDAToolkit_LIBRARY_ROOT}/lib/stubs")
        endif()
      endif()

      list(APPEND _legion_cuda_options "CUDA_NVCC_FLAGS ${_nvcc_flags}")
      list(APPEND _legion_cuda_options "CMAKE_LIBRARY_PATH ${_lib_path}")
      list(APPEND _legion_cuda_options "CMAKE_CUDA_STANDARD ${_cuda_std}")
    endif()

    # Because legion sets these as cache variables, we need to force set this as a cache variable here
    # to ensure that Legion doesn't override this in the CMakeCache.txt and create an unexpected state.
    # This only applies to set() but does not apply to option() variables.
    # See discussion of FetchContent subtleties:
    # Only use these FORCE calls if using a Legion subbuild.
    # https://discourse.cmake.org/t/fetchcontent-cache-variables/1538/8
    set(Legion_MAX_DIM ${Legion_MAX_DIM} CACHE STRING "The max number of dimensions for Legion" FORCE)
    set(Legion_MAX_FIELDS ${Legion_MAX_FIELDS} CACHE STRING "The max number of fields for Legion" FORCE)

    rapids_cpm_find(Legion ${PKG_VERSION} ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legion_cpm_git_args}
          FIND_PACKAGE_ARGUMENTS EXACT
          EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
          OPTIONS                ${_legion_cuda_options}
                                 "CMAKE_CXX_STANDARD ${_cxx_std}"
                                 "Legion_VERSION ${PKG_VERSION}"
                                 "Legion_BUILD_BINDINGS ON"
                                 "Legion_BUILD_APPS OFF"
                                 "Legion_BUILD_TESTS OFF"
                                 "Legion_BUILD_TUTORIAL OFF"
                                 "Legion_REDOP_HALF ON"
                                 "Legion_REDOP_COMPLEX ON"
                                 "Legion_GPU_REDUCTIONS OFF"
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_USE_Python ${Legion_USE_Python} PARENT_SCOPE)
  if("${Legion_NETWORKS}" MATCHES ".*gasnet(1|ex).*")
    set(Legion_USE_GASNet ON PARENT_SCOPE)
  endif()

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_USE_Python=${Legion_USE_Python}")
  message(VERBOSE "Legion_USE_GASNet=${Legion_USE_GASNet}")

endfunction()

if(NOT DEFINED legate_core_LEGION_BRANCH)
  set(legate_core_LEGION_BRANCH control_replication)
endif()

if(NOT DEFINED legate_core_LEGION_REPOSITORY)
  set(legate_core_LEGION_REPOSITORY https://gitlab.com/StanfordLegion/legion.git)
endif()

if(NOT DEFINED legate_core_LEGION_VERSION)
  set(legate_core_LEGION_VERSION "${legate_core_VERSION_MAJOR}.${legate_core_VERSION_MINOR}.0")
endif()

find_or_configure_legion(VERSION          ${legate_core_LEGION_VERSION}
                         REPOSITORY       ${legate_core_LEGION_REPOSITORY}
                         BRANCH           ${legate_core_LEGION_BRANCH}
                         EXCLUDE_FROM_ALL ${legate_core_EXCLUDE_LEGION_FROM_ALL}
)
