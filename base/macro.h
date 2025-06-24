#pragma once

#define ORDER 1
#define NSP (ORDER + 1)

#if USE_DOUBLE_PRECISION
using DataType = double; // 双精度类型
#else
using DataType = float; // 单精度类型
#endif