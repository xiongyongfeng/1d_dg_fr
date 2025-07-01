
#pragma once
#include "config.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace nlohmann
{

template <>
struct adl_serializer<Config>
{
    static void to_json(json &j, const Config &c)
    {
        j = json{{"x0", c.x0},       {"x1", c.x1},
                 {"n_ele", c.n_ele}, {"dt", c.dt},
                 {"n_dt", c.n_dt},   {"out_time_step", c.out_time_step},
                 {"a", c.a},         {"output_dir", c.output_dir}};
    }
    static void from_json(const json &j, Config &c)
    {
        j.at("x0").get_to(c.x0);
        j.at("x1").get_to(c.x1);
        j.at("n_ele").get_to(c.n_ele);
        j.at("dt").get_to(c.dt);
        j.at("n_dt").get_to(c.n_dt);
        j.at("out_time_step").get_to(c.out_time_step);
        j.at("a").get_to(c.a);
        j.at("output_dir").get_to(c.output_dir);
    }
};

Config loadConfig(const std::string &filename);

void saveConfig(const Config &config, const std::string &filename);

} // namespace nlohmann