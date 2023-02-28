# We abuse find package for testing purposes here to
# 'find' the current build tree to test package builds
set(legate_core_ROOT ${CMAKE_BINARY_DIR})

add_subdirectory(hello)
