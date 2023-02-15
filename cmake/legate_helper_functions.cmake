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

function(legate_add_cffi header)
execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E
    -P "${header}"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header_content
  COMMAND_ERROR_IS_FATAL ANY
)

set(install_info_in
[=[
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
        target = "@target"
        if exists(join(libdir, f"lib{target}{so_ext}")):
            return libdir
        return None

    return (
        find_libcunumeric(join(cn_path, "build", "lib")) or
        find_libcunumeric(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_libcunumeric(join(dirname(dirname(sys.executable)), "lib")) or
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
