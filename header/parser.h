
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
        j = json{{"x0", c.x0},
                 {"x1", c.x1},
                 {"n_ele", c.n_ele},
                 {"dt", c.dt},
                 {"total_time", c.total_time},
                 {"output_time_step", c.output_time_step},
                 {"a", c.a},
                 {"output_dir", c.output_dir},
                 {"limiter_type", c.limiter_type},
                 {"enable_entropy_modify", c.enable_entropy_modify},
                 {"dg_fr_type", c.dg_fr_type}};
    }
    static void from_json(const json &j, Config &c)
    {
        j.at("x0").get_to(c.x0);
        j.at("x1").get_to(c.x1);
        j.at("n_ele").get_to(c.n_ele);
        j.at("dt").get_to(c.dt);
        j.at("total_time").get_to(c.total_time);
        j.at("output_time_step").get_to(c.output_time_step);
        j.at("a").get_to(c.a);
        j.at("output_dir").get_to(c.output_dir);
        auto it = j.find("limiter_type");
        if (it != j.end())
        {
            it->get_to(c.limiter_type);
        }
        it = j.find("enable_entropy_modify");
        if (it != j.end())
        {
            it->get_to(c.enable_entropy_modify);
        }
        it = j.find("dg_fr_type");
        if (it != j.end())
        {
            it->get_to(c.dg_fr_type);
        }
    }
};

Config loadConfig(const std::string &filename);

void saveConfig(const Config &config, const std::string &filename);

} // namespace nlohmann