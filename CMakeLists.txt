cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20) # 设置全局 C++ 标准为 17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求编译器支持 C++17，否则报错
set(CMAKE_CXX_EXTENSIONS OFF) # 禁用编译器特有扩展（如 GNU 的 -std=gnu++17
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(1d_dg_fr)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

set(TARGET_NAME main_ns)
set(TARGET_LAD main_lad)


file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE EXCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/build/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dist/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/lad.cpp"
)
list(REMOVE_ITEM SOURCE_FILES ${EXCLUDE_FILES})
message(STATUS "SOURCE_FILES=${SOURCE_FILES}")

add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_include_directories(${TARGET_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)
target_compile_definitions(${TARGET_NAME} PRIVATE NS=1)


file(GLOB_RECURSE SOURCE_FILES_LAD
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE EXCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/build/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dist/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/ns.cpp"
)
list(REMOVE_ITEM SOURCE_FILES_LAD ${EXCLUDE_FILES})
message(STATUS "SOURCE_FILES_LAD=${SOURCE_FILES_LAD}")
add_executable(${TARGET_LAD} ${SOURCE_FILES_LAD})
target_include_directories(${TARGET_LAD}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)
target_compile_definitions(${TARGET_LAD} PRIVATE LAD=1)