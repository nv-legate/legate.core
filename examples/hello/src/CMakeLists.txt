
legate_cpp_library_template(hello TEMPLATE_SOURCES)

add_library(
  hello
  hello_world.cc
  hello_world.h
  iota.cc
  sum.cc
  square.cc
  ${TEMPLATE_SOURCES}
)

target_include_directories(hello
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hello PRIVATE legate::core)

