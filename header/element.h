#pragma once
#include "constants.h"
#include "macro.h"

struct Geom
{
    DataType x[NSP];
};

struct Element
{
    DataType u_consrv[NSP];
    DataType u_grad_consrv[NSP];
};

struct Rhs
{
    DataType rhs[NSP];
};