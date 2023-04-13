legate_cpp_library_template(collective TEMPLATE_SOURCES)

add_library(
  collective
  collective_test.cc
  collective_test.h
  ${TEMPLATE_SOURCES}
)

target_include_directories(collective
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(collective PRIVATE legate::core)
