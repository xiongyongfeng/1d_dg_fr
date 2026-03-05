#pragma once
#include "macro.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

enum class BcType
{
    Periodic = 0,
    Dirichlet = 1
};

struct Config
{
    DataType x0;
    DataType x1;
    int n_ele;
    DataType total_time;
    DataType output_time_step;
    DataType a; // linear advection coef
    DataType nu;// linear diffusion coef
    DataType cfl = 0.4; // CFL数，如果 > 0 则自动计算dt
    DataType dt = 1e-8;
    std::string output_dir;
    int limiter_type = 0; // 0 no limiter, 1 tvd limiter
    int dg_fr_type = 0;   // 0 for DG, 1 for FR
    bool enable_entropy_modify = false;
    DataType weight = 0.5;
    int time_scheme_type = 0; // 0 for tvd-rk, 1 for new time scheme
    int vis_scheme_type = 1; // 0 for br2, 1 for ipdg
    DataType ip_coef = 1.0; // 内部惩罚系数，IPDG方法使用
    int bc_type = 0; // 0 for periodic, 1 for dirichlet
    DataType bc_left = 0.0;  // Dirichlet边界条件: 左边界值
    DataType bc_right = 0.0; // Dirichlet边界条件: 右边界值
};
