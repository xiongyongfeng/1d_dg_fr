#pragma once
#include "constants.h"
#include "macro.h"

struct Element
{
    DataType x[NSP];
    DataType u_consrv[NSP];
    DataType u_grad_consrv[NSP];
    DataType rhs[NSP];
};