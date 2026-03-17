#pragma once
#include <cstddef>
#include <iostream>
// #define USE_DOUBLE_PRECISION 1
#if USE_DOUBLE_PRECISION
using DataType = double; // 双精度类型
#else
using DataType = float; // 单精度类型
#endif
constexpr size_t ORDER = 1;
constexpr size_t NSP = ORDER + 1;

constexpr size_t TORDER = 1;
constexpr size_t NTP = TORDER + 1;
#ifdef NS
constexpr size_t NCONSRV = 3;
constexpr size_t NPRIMTV = 4;
#endif

#ifdef LAD
constexpr size_t NCONSRV = 1;
constexpr size_t NPRIMTV = 1;
#endif

#ifdef BURGERS
constexpr size_t NCONSRV = 1;
constexpr size_t NPRIMTV = 1;
#endif

constexpr DataType GAMMA = 1.4;
constexpr DataType GAS_R = 1.0;