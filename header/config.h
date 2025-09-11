#pragma once
#include "macro.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

struct Config
{
    DataType x0;
    DataType x1;
    int n_ele;
    DataType dt;
    DataType total_time;
    DataType output_time_step;
    DataType a; // linear advection coef
    std::string output_dir;
    int limiter_type = 0; // 0 no limiter, 1 tvd limiter
    int dg_fr_type = 0;   // 0 for DG, 1 for FR
    bool enable_entropy_modify = false;
    DataType weight = 0.5;
    int time_scheme_type = 0; // 0 for tvd-rk, 1 for new time scheme
};