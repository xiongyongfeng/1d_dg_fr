#pragma once
#include "macro.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

struct SolverConfig {
  DataType x0;
  DataType x1;
  int n_ele;
};