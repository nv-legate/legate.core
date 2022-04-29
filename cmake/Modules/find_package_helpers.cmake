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

function(set_package_dir_if_built)
  set(oneValueArgs NAME)
  set(multiValueArgs BUILD_ARTIFACTS)

  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # respect external <pkg>_DIR or <pkg>_ROOT if set
  if(${PKG_NAME}_DIR OR ${PKG_NAME}_ROOT)
    return()
  endif()

  # respect external FETCHCONTENT_BASE_DIR if set
  if(DEFINED FETCHCONTENT_BASE_DIR)
    set(CPM_FETCHCONTENT_BASE_DIR ${FETCHCONTENT_BASE_DIR})
  else()
    set(CPM_FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps)
  endif()

  string(TOLOWER ${PKG_NAME} pkg_name)
  set(build_dir "${CPM_FETCHCONTENT_BASE_DIR}/${pkg_name}-build")

  foreach(artifact IN LISTS PKG_BUILD_ARTIFACTS)
    if(NOT EXISTS "${build_dir}/${artifact}")
      return()
    endif()
  endforeach()

  message(VERBOSE "setting ${PKG_NAME}_DIR to '${build_dir}'")

  set(${PKG_NAME}_DIR "${build_dir}" PARENT_SCOPE)
endfunction()
