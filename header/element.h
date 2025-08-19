#pragma once
#include "constants.h"
#include "macro.h"
#include <cstdint>

struct Geom
{
    DataType x[NSP];
    DataType dx;
    DataType local_det_jac;
};

struct Element
{
    DataType u_consrv[NSP][NCONSRV];
    DataType u_avg[NCONSRV];
    DataType u_grad_consrv[NSP][NCONSRV];
    int32_t islimited[NCONSRV];
};

struct Rhs
{
    DataType rhs[NSP][NCONSRV];
};