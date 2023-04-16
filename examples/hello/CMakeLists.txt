#=============================================================================
# Copyright 2023 NVIDIA Corporation
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

project(hello VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

find_package(legate_core REQUIRED)

legate_cpp_library_template(src TARGET hello)
legate_add_cpp_subdirectory(src TARGET hello EXPORT hello-export)

legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/hello_cffi.h TARGET hello)
legate_python_library_template(hello)
legate_default_python_install(hello EXPORT hello-export)
