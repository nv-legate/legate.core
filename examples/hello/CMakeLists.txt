cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(hello VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

find_package(legate_core REQUIRED)

legate_add_cpp_subdirectory(src TARGET hello EXPORT hello-export)

legate_add_cffi(${CMAKE_CURRENT_SOURCE_DIR}/src/hello_cffi.h TARGET hello)
legate_python_library_template(hello)
legate_default_python_install(hello EXPORT hello-export)
