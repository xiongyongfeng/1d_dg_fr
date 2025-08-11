#pragma once
#include "constants.h"
#include "macro.h"
#include <cstdint>

struct Geom
{
    DataType x[NSP];
};

struct Element
{
    DataType u_consrv[NSP];
    DataType u_avg;
    DataType u_grad_consrv[NSP];
    int32_t islimited;
};

struct Rhs
{
    DataType rhs[NSP];
};