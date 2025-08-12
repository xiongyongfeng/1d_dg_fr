#include "parser.h"
#include <iostream>

namespace nlohmann
{

Config loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("配置文件打开失败");
    nlohmann::json j;
    file >> j; // 解析JSON文件
    std::cout << j.dump(4);
    return j.get<Config>(); // 自动反序列化到结构体
}

void saveConfig(const Config &config, const std::string &filename)
{
    nlohmann::json j = config; // 自动序列化
    std::ofstream file(filename);
    file << j.dump(4); // 缩进4格输出，便于阅读
}
} // namespace nlohmann