legate_cpp_library_template(integration TEMPLATE_SOURCES)

add_library(
  integration
  #mapper.h
  #mapper.cc
  collective_test.cc
  collective_test.h
  ${TEMPLATE_SOURCES}
)

target_include_directories(integration
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(integration PRIVATE legate::core)
