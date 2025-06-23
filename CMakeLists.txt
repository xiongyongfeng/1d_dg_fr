cmake_minimum_required(VERSION 3.15)
project(1d_dg_fr)
add_executable(main main.cpp)
target_compile_definitions(main PRIVATE USE_DOUBLE_PRECISION=1)