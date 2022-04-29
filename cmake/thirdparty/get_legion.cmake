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
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(Legion_CUDA_ARCH "")
  if(Legion_USE_CUDA)
    set(Legion_CUDA_ARCH ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-real" "")
    list(TRANSFORM Legion_CUDA_ARCH REPLACE "-virtual" "")
    list(JOIN Legion_CUDA_ARCH "," Legion_CUDA_ARCH)
  endif()

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/find_package_helpers.cmake)
  set_package_dir_if_built(NAME            Legion
                           BUILD_ARTIFACTS lib/librealm.so
                                           lib/libregent.so
                                           lib/liblegion.so)

  rapids_cpm_find(Legion  ${PKG_VERSION}
      GLOBAL_TARGETS      Legion::Realm
                          Legion::Regent
                          Legion::Legion
                          Legion::RealmRuntime
                          Legion::LegionRuntime
      BUILD_EXPORT_SET    legate-core-exports
      INSTALL_EXPORT_SET  legate-core-exports
      CPM_ARGS
        GIT_REPOSITORY   ${PKG_REPOSITORY}
        GIT_TAG          ${PKG_PINNED_TAG}
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS          "CMAKE_CXX_STANDARD 17"
                         "Legion_BUILD_BINDINGS ON"
                         "Legion_BUILD_APPS OFF"
                         "Legion_BUILD_TESTS OFF"
                         "Legion_BUILD_TUTORIAL OFF"
                         "Legion_REDOP_HALF ON"
                         "Legion_REDOP_COMPLEX ON"
                         "Legion_GPU_REDUCTIONS OFF"
                         "Legion_CUDA_ARCH ${Legion_CUDA_ARCH}"
  )

endfunction()

if(NOT DEFINED LEGATE_CORE_LEGION_BRANCH)
  set(LEGATE_CORE_LEGION_BRANCH control_replication)
endif()

if(NOT DEFINED LEGATE_CORE_LEGION_REPOSITORY)
  set(LEGATE_CORE_LEGION_REPOSITORY https://gitlab.com/StanfordLegion/legion.git)
endif()

set(LEGATE_CORE_Legion_MIN_VERSION "${LEGATE_CORE_VERSION_MAJOR}.${LEGATE_CORE_VERSION_MINOR}")

find_or_configure_legion(VERSION          "${LEGATE_CORE_Legion_MIN_VERSION}.0"
                         REPOSITORY       ${LEGATE_CORE_LEGION_REPOSITORY}
                         PINNED_TAG       ${LEGATE_CORE_LEGION_BRANCH}
                         EXCLUDE_FROM_ALL ${LEGATE_CORE_EXCLUDE_LEGION_FROM_ALL}
)
