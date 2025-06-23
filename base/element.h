#pragma once
#include "constants.h"
#include "macro.h"

struct Element
{
    DataType u_consrv[NSP];
    DataType u_grad_consrv[NSP];
    DataType x[NSP];
};