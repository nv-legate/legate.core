macro(legate_include_rapids)
  if (NOT _LEGATE_HAS_RAPIDS)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
      file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
           ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    endif()
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
endmacro()

function(legate_default_cpp_install target)
  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  legate_include_rapids()

  rapids_cmake_install_lib_dir(lib_dir)

  install(TARGETS ${target}
          DESTINATION ${lib_dir}
	  EXPORT ${OPT_EXPORT})

  set(code_string "set(legate_core_DIR ${legate_core_DIR})")

  rapids_export(
    INSTALL ${target}
    EXPORT_SET ${OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    FINAL_CODE_BLOCK code_string
  )

  # build export targets
  rapids_export(
    BUILD ${target}
    EXPORT_SET ${OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    FINAL_CODE_BLOCK code_string
  )
endfunction()

function(legate_add_cffi header target)
execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E
    -P "${header}"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header_output
  COMMAND_ERROR_IS_FATAL ANY
)
string(JOIN "\n" header_content
  "${header_output}"
  "void ${target}_perform_registration();"
)

set(install_info_in
[=[
from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"libhello{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
	find_lib("@CMAKE_BINARY_DIR@") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()
header: str = """@header_content@"""
]=])

set(libpath "")
string(CONFIGURE "${install_info_in}" install_info @ONLY)
file(WRITE ${CMAKE_SOURCE_DIR}/${target}/install_info.py "${install_info}")
endfunction()

function(legate_default_python_install target)
  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_python_install")
  endif()

  if (SKBUILD)
    add_library(${target}_python INTERFACE)
    add_library(legate::${target}_python ALIAS ${target}_python)
    target_link_libraries(${target}_python INTERFACE legate::core legate::${target})

    install(TARGETS ${target}_python
            DESTINATION ${lib_dir}
            EXPORT ${OPT_EXPORT})

    legate_include_rapids()
    rapids_export(
      INSTALL ${target}_python
      EXPORT_SET ${OPT_EXPORT}
      GLOBAL_TARGETS ${target}_python
      NAMESPACE legate::
    )
  endif()
endfunction()

function(legate_add_cpp_subdirectory dir target)
  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  legate_include_rapids()

  rapids_find_package(legate_core CONFIG
          GLOBAL_TARGETS legate::core
          BUILD_EXPORT_SET ${OPT_EXPORT}
          INSTALL_EXPORT_SET ${OPT_EXPORT})

  if (SKBUILD)
    if (NOT DEFINED ${target}_ROOT)
      set(${target}_ROOT ${CMAKE_SOURCE_DIR}/build)
    endif()
    rapids_find_package(${target} CONFIG
      GLOBAL_TARGETS legate::${target}
      BUILD_EXPORT_SET ${OPT_EXPORT}
      INSTALL_EXPORT_SET ${OPT_EXPORT})
    if (NOT ${target}_FOUND)
      add_subdirectory(${dir} ${CMAKE_SOURCE_DIR}/build)
      legate_default_cpp_install(${target} EXPORT ${OPT_EXPORT})
    endif()
  else()
    add_subdirectory(${dir})
    legate_default_cpp_install(${target} EXPORT ${OPT_EXPORT})
  endif()

endfunction()

function(legate_cpp_library_template target output_sources_variable)
  set(file_template
[=[
#pragma once

#include "legate.h"

namespace @target@ {

struct Registry {
 public:
  template <typename... Args>
  static void record_variant(Args&&... args)
  {
    get_registrar().record_variant(std::forward<Args>(args)...);
  }
  static legate::LegateTaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar = Registry;
  static constexpr int TASK_ID = ID;
};

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.h "${file_content}")
 
  set(file_template
[=[
#include "legate_library.h"
#include "core/mapping/base_mapper.h"

namespace @target@ {

class Mapper : public legate::mapping::BaseMapper {

 public:
  Mapper(Legion::Runtime* rt, Legion::Machine machine, const legate::LibraryContext& context)
  : BaseMapper(rt, machine, context)
  {
  }

  virtual ~Mapper(void) {}

 private:
  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  bool is_pure() const override { return true; }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override {
    return *options.begin();
  }
 
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override {
    using legate::mapping::StoreMapping;
    std::vector<StoreMapping> mappings;
    auto& inputs  = task.inputs();
    auto& outputs = task.outputs();
    for (auto& input : inputs) {
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.exact = true;
    }
    for (auto& output : outputs) {
      mappings.push_back(StoreMapping::default_mapping(output, options.front()));
      mappings.back().policy.exact = true;
    }
    return std::move(mappings);
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override {
    return 0;
  }
};

static const char* const library_name = "hello";

Legion::Logger log_hello("hello");

/*static*/ legate::LegateTaskRegistrar& Registry::get_registrar()
{
  static legate::LegateTaskRegistrar registrar;
  return registrar;
}

void registration_callback(Legion::Machine machine,
                           Legion::Runtime* runtime,
                           const std::set<Legion::Processor>& local_procs)
{
  legate::ResourceConfig config;
  config.max_mappers       = 1;
  config.max_tasks         = 1024;
  config.max_reduction_ops = 8;
  legate::LibraryContext context(runtime, library_name, config);

  Registry::get_registrar().register_all_tasks(runtime, context);

  // Now we can register our mapper with the runtime
  context.register_mapper(new Mapper(runtime, machine, context), 0);
}

}  // namespace @target@

extern "C" {

void @target@_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  Legion::Runtime::perform_registration_callback(@target@::registration_callback, true /*global*/);
}

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.cc "${file_content}")

  set(${output_sources_variable} 
    legate_library.h
    legate_library.cc
    PARENT_SCOPE
  )
endfunction()

function(legate_python_library_template target)
set(file_template
[=[
from legate.core import (
    Library,
    ResourceConfig,
    get_legate_runtime,
)
import os
from typing import Any

class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from hello.install_info import libpath
        return os.path.join(libpath, f"lib@target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from hello.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "hello_perform_registration"

    def get_resource_configuration(self) -> ResourceConfig:
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_tasks = 1024
        config.max_mappers = 1
        config.max_reduction_ops = 8
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

user_lib = UserLibrary("@target@")
user_context = get_legate_runtime().register_library(user_lib)
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_SOURCE_DIR}/${target}/library.py "${file_content}")
endfunction()
