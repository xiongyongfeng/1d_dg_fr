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
};