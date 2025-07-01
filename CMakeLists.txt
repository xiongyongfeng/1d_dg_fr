cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20) # 设置全局 C++ 标准为 17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求编译器支持 C++17，否则报错
set(CMAKE_CXX_EXTENSIONS OFF) # 禁用编译器特有扩展（如 GNU 的 -std=gnu++17
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(1d_dg_fr)

set(TARGET_NAME main)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE EXCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/build/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dist/*.cpp" # 可扩展其他目录
)

# 从主列表中移除排除文件
list(REMOVE_ITEM SOURCE_FILES ${EXCLUDE_FILES})
add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_compile_definitions(${TARGET_NAME} PRIVATE USE_DOUBLE_PRECISION=1)
target_include_directories(${TARGET_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/header
)

if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    message(WARNING "Source directory 'examples' does not exist!")
else()
    add_custom_command(
        TARGET ${TARGET_NAME} # 绑定到主目标（如可执行文件）
        POST_BUILD # 在主目标构建完成后执行
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/examples
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${CMAKE_CURRENT_BINARY_DIR}/examples
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples
        COMMENT "Copying example directory to build folder..."
    )
endif()