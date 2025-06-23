
#pragma once
#include "solver_config.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace nlohmann {

template <> struct adl_serializer<SolverConfig> {
  static void to_json(json &j, const SolverConfig &c) {
    j = json{{"x0", c.x0}, {"x1", c.x1}, {"n_ele", c.n_ele}};
  }
  static void from_json(const json &j, SolverConfig &c) {
    j.at("x0").get_to(c.x0);
    j.at("x1").get_to(c.x1);
    j.at("n_ele").get_to(c.n_ele);
  }
};

SolverConfig loadConfig(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("配置文件打开失败");
  nlohmann::json j;
  file >> j;                    // 解析JSON文件
  return j.get<SolverConfig>(); // 自动反序列化到结构体
}

void saveConfig(const SolverConfig &config, const std::string &filename) {
  nlohmann::json j = config; // 自动序列化
  std::ofstream file(filename);
  file << j.dump(4); // 缩进4格输出，便于阅读
}

} // namespace nlohmann