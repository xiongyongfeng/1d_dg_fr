cmake_minimum_required(VERSION 3.15)
project(1d_dg_fr)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

file(GLOB_RECURSE SOURCE_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE EXCLUDE_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/build/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dist/*.cpp"  # 可扩展其他目录
)
# 从主列表中移除排除文件
list(REMOVE_ITEM SOURCE_FILES ${EXCLUDE_FILES})
add_executable(main ${SOURCE_FILES})
message("SOURCE_FILES=${SOURCE_FILES}")
target_compile_definitions(main PRIVATE USE_DOUBLE_PRECISION=1)