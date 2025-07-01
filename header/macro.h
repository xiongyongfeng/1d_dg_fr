#pragma once
#include <iostream>

constexpr size_t ORDER = 1;
constexpr size_t NSP = ORDER + 1;

#if USE_DOUBLE_PRECISION
using DataType = double; // 双精度类型
#else
using DataType = float; // 单精度类型
#endif