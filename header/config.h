#pragma once
#include "macro.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

enum class BcType
{
    Periodic = 0,
    Dirichlet = 1,
    Symmetry = 2
};

enum class CommonFluxType
{
    LF = 0,
    HLL = 1,
    HLLC = 2,
    ROE = 3,
    AUSM = 4,
    AUSM_PLUS_UP = 5,
};

/**
 * @brief 配置基类 - 包含所有求解器共有的参数
 */
struct ConfigBase
{
    // 网格参数
    DataType x0;
    DataType x1;
    int n_ele;

    // 时间参数
    DataType total_time;
    DataType output_time_step;
    DataType cfl = 0.4; // CFL数，如果 > 0 则自动计算dt
    DataType dt = 1e-8;

    // 输出参数
    std::string output_dir;

    // 数值方法参数
    int limiter_type = 0; // 0 no limiter, 1 tvd limiter
    int dg_fr_type = 0;   // 0 for DG, 1 for FR
    bool enable_entropy_modify = false;
    DataType weight = 0.5;
    int time_scheme_type = 0; // 0 for tvd-rk3, 1 for new time scheme

    // 边界条件参数
    BcType bc_type = BcType::Periodic;
    DataType bc_left = 0.0;  // Dirichlet边界条件: 左边界值
    DataType bc_right = 0.0; // Dirichlet边界条件: 右边界值
};

/**
 * @brief LAD (Linear Advection Diffusion) 求解器配置
 * u_t + a * u_x = nu * u_xx
 */
struct ConfigLAD : public ConfigBase
{
    // LAD特有参数
    DataType a;              // linear advection coefficient
    DataType nu;             // linear diffusion coefficient
    int vis_scheme_type = 1; // 0 for br2, 1 for ipdg
    DataType ip_coef = 1.0;  // 内部惩罚系数，IPDG方法使用
};

/**
 * @brief Burgers 求解器配置
 * u_t + (0.5 * u^2)_x = 0
 */
struct ConfigBurgers : public ConfigBase
{
    // Burgers目前无特有参数
    // 预留扩展空间
};

/**
 * @brief NS (Euler/Navier-Stokes) 求解器配置
 * 一维Euler方程
 */
struct ConfigNS : public ConfigBase
{
    // NS目前无特有参数（GAMMA等在constants.h中定义）
    // 预留扩展空间
    CommonFluxType common_flux_type = CommonFluxType::HLL;
};