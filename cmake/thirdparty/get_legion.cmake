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

  set(Legion_CUDA_ARCH "")
  if(Legion_USE_CUDA)
    set(Legion_CUDA_ARCH ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-real" "")
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-virtual" "")
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
      set(Legion_PYTHON_EXTRA_INSTALL_ARGS "--single-version-externally-managed --root=/")
    endif()

    set(_cuda_path "")
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

      # Get the `stubs/libcuda.so` path so we can set CMAKE_LIBRARY_PATH for FindCUDA.cmake
      set(_libdir "lib64")
      if(CMAKE_SIZEOF_VOID_P LESS 8)
        set(_libdir "lib")
      endif()

      if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/stubs/libcuda.so")
        # This might be the path to the `$CONDA_PREFIX/lib`
        # If it is (and it has the libcuda.so driver stub),
        # then we know we're using the cuda-toolkit package
        # and should link to that driver stub instead of the
        # one potentially in `/usr/local/cuda/lib[64]/stubs`
        list(APPEND _cuda_stubs "${CUDAToolkit_LIBRARY_DIR}/stubs")
      elseif(EXISTS "${CUDAToolkit_TARGET_DIR}/${_libdir}/stubs/libcuda.so")
        # Otherwise assume stubs are relative to the CUDA toolkit root dir
        list(APPEND _cuda_stubs "${CUDAToolkit_TARGET_DIR}/${_libdir}/stubs")
      elseif(EXISTS "${CUDAToolkit_LIBRARY_ROOT}/${_libdir}/stubs/libcuda.so")
        list(APPEND _cuda_stubs "${CUDAToolkit_LIBRARY_ROOT}/${_libdir}/stubs")
      elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/${_libdir}/stubs/libcuda.so")
        # Use CUDA_PATH envvar (if set)
        list(APPEND _cuda_stubs "$ENV{CUDA_PATH}/${_libdir}/stubs/libcuda.so")
      elseif(DEFINED ENV{CUDA_LIB_PATH} AND EXISTS "$ENV{CUDA_LIB_PATH}/stubs/libcuda.so")
        # Use CUDA_LIB_PATH envvar (if set)
        list(APPEND _cuda_stubs "$ENV{CUDA_LIB_PATH}/stubs/libcuda.so")
      elseif(DEFINED ENV{LIBRARY_PATH} AND
            ("$ENV{LIBRARY_PATH}" STREQUAL "/usr/local/cuda/${_libdir}/stubs"))
        # LIBRARY_PATH is set in the `nvidia/cuda` containers to /usr/local/cuda/lib64/stubs
        list(APPEND _cuda_stubs "$ENV{LIBRARY_PATH}")
      else()
        message(FATAL_ERROR "Could not find the libcuda.so driver stub. "
                            "Please reconfigure with -DCUDAToolkit_ROOT= "
                            "set to a valid CUDA Toolkit installation.")
      endif()

      message(VERBOSE "legate.core: Path(s) to CUDA stubs: ${_cuda_stubs}")

      list(APPEND _legion_cuda_options "CUDA_NVCC_FLAGS ${_nvcc_flags}")
      list(APPEND _legion_cuda_options "CMAKE_CUDA_STANDARD ${_cuda_std}")
      # Set this so Legion correctly finds the CUDA toolkit.
      list(APPEND _legion_cuda_options "CMAKE_LIBRARY_PATH ${_cuda_stubs}")

      # Set these as cache variables for the legacy FindCUDA.cmake
      set(CUDA_VERBOSE_BUILD ON CACHE BOOL "" FORCE)
      set(CUDA_USE_STATIC_CUDA_RUNTIME ${legate_core_STATIC_CUDA_RUNTIME} CACHE BOOL "" FORCE)

      # Ensure `${_cuda_stubs}/libcuda.so` doesn't end up in the RPATH of the legion_python binary
      list(APPEND CMAKE_C_IMPLICIT_LINK_DIRECTORIES "${_cuda_stubs}")
      list(APPEND CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "${_cuda_stubs}")
      list(APPEND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${_cuda_stubs}")
      set(legate_core_cuda_stubs_path "${_cuda_stubs}" PARENT_SCOPE)
      set(legate_core_cuda_stubs_path "${_cuda_stubs}" CACHE STRING "" FORCE)
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
    set(Legion_CUDA_ARCH ${Legion_CUDA_ARCH} CACHE STRING
      "Comma-separated list of CUDA architectures to build for (e.g. 60,70)" FORCE)

    message(VERBOSE "legate.core: Legion version: ${version}")
    message(VERBOSE "legate.core: Legion git_repo: ${git_repo}")
    message(VERBOSE "legate.core: Legion git_branch: ${git_branch}")
    message(VERBOSE "legate.core: Legion exclude_from_all: ${exclude_from_all}")

    rapids_cpm_find(Legion ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          ${legion_cpm_git_args}
          FIND_PACKAGE_ARGUMENTS EXACT
          EXCLUDE_FROM_ALL       ${exclude_from_all}
          OPTIONS                ${_legion_cuda_options}
                                 "CMAKE_CXX_STANDARD ${_cxx_std}"
                                 "Legion_VERSION ${version}"
                                 "Legion_BUILD_BINDINGS ON"
                                 "Legion_BUILD_APPS OFF"
                                 "Legion_BUILD_TESTS OFF"
                                 "Legion_BUILD_TUTORIAL OFF"
                                 "Legion_REDOP_HALF ON"
                                 "Legion_REDOP_COMPLEX ON"
                                 "Legion_GPU_REDUCTIONS OFF"
                                 "Legion_BUILD_RUST_PROFILER ON"
                                 "Legion_SPY ${Legion_SPY}"
                                 "Legion_USE_LLVM ${Legion_USE_LLVM}"
                                 "Legion_USE_HDF5 ${Legion_USE_HDF5}"
                                 "Legion_USE_CUDA ${Legion_USE_CUDA}"
                                 "Legion_NETWORKS ${Legion_NETWORKS}"
                                 "Legion_USE_OpenMP ${Legion_USE_OpenMP}"
                                 "Legion_USE_Python ${Legion_USE_Python}"
                                 "Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS}"
    )
  endif()

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_USE_Python ${Legion_USE_Python} PARENT_SCOPE)
  set(Legion_NETWORKS ${Legion_NETWORKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_USE_Python=${Legion_USE_Python}")
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
