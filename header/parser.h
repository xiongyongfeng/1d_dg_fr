#pragma once
#include "config.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace nlohmann
{

// ConfigBase 序列化
template <>
struct adl_serializer<ConfigBase>
{
    static void to_json(ordered_json &j, const ConfigBase &c)
    {
        j = ordered_json{{"x0", c.x0},
                 {"x1", c.x1},
                 {"n_ele", c.n_ele},
                 {"total_time", c.total_time},
                 {"output_time_step", c.output_time_step},
                 {"cfl", c.cfl},
                 {"dt", c.dt},
                 {"output_dir", c.output_dir},
                 {"limiter_type", c.limiter_type},
                 {"dg_fr_type", c.dg_fr_type},
                 {"enable_entropy_modify", c.enable_entropy_modify},
                 {"weight", c.weight},
                 {"time_scheme_type", c.time_scheme_type},
                 {"bc_type", c.bc_type},
                 {"bc_left", c.bc_left},
                 {"bc_right", c.bc_right}};
    }
    static void from_json(const ordered_json &j, ConfigBase &c)
    {
        j.at("x0").get_to(c.x0);
        j.at("x1").get_to(c.x1);
        j.at("n_ele").get_to(c.n_ele);
        j.at("total_time").get_to(c.total_time);
        j.at("output_time_step").get_to(c.output_time_step);
        j.at("output_dir").get_to(c.output_dir);

        auto it = j.find("dt");
        if (it != j.end())
        {
            it->get_to(c.dt);
        }

        it = j.find("cfl");
        if (it != j.end())
        {
            it->get_to(c.cfl);
        }

        it = j.find("limiter_type");
        if (it != j.end())
        {
            it->get_to(c.limiter_type);
        }

        it = j.find("enable_entropy_modify");
        if (it != j.end())
        {
            it->get_to(c.enable_entropy_modify);
        }

        it = j.find("weight");
        if (it != j.end())
        {
            it->get_to(c.weight);
        }

        it = j.find("time_scheme_type");
        if (it != j.end())
        {
            it->get_to(c.time_scheme_type);
        }

        it = j.find("dg_fr_type");
        if (it != j.end())
        {
            it->get_to(c.dg_fr_type);
        }

        it = j.find("bc_type");
        if (it != j.end())
        {
            it->get_to(c.bc_type);
        }
        else
        {
            c.bc_type = BcType::Periodic;
        }

        it = j.find("bc_left");
        if (it != j.end())
        {
            it->get_to(c.bc_left);
        }
        else
        {
            c.bc_left = 0.0;
        }

        it = j.find("bc_right");
        if (it != j.end())
        {
            it->get_to(c.bc_right);
        }
        else
        {
            c.bc_right = 0.0;
        }
    }
};

// ConfigLAD 序列化
template <>
struct adl_serializer<ConfigLAD>
{
    static void to_json(ordered_json &j, const ConfigLAD &c)
    {
         adl_serializer<ConfigBase>::to_json(j,static_cast<const ConfigBase&>(c));
        j["a"] = c.a;
        j["nu"] = c.nu;
        j["vis_scheme_type"] = c.vis_scheme_type;
        j["ip_coef"] = c.ip_coef;
    }
    static void from_json(const ordered_json &j, ConfigLAD &c)
    {
        adl_serializer<ConfigBase>::from_json(j, static_cast<ConfigBase&>(c));
        j.at("a").get_to(c.a);

        auto it = j.find("nu");
        if (it != j.end())
        {
            it->get_to(c.nu);
        }
        else
        {
            c.nu = 0.0;
        }

        it = j.find("vis_scheme_type");
        if (it != j.end())
        {
            it->get_to(c.vis_scheme_type);
        }

        it = j.find("ip_coef");
        if (it != j.end())
        {
            it->get_to(c.ip_coef);
        }
    }
};

// ConfigBurgers 序列化
template <>
struct adl_serializer<ConfigBurgers>
{
    static void to_json(ordered_json &j, const ConfigBurgers &c)
    {
         adl_serializer<ConfigBase>::to_json(j,static_cast<const ConfigBase&>(c));
    }
    static void from_json(const ordered_json &j, ConfigBurgers &c)
    {
        adl_serializer<ConfigBase>::from_json(j, static_cast<ConfigBase&>(c));
    }
};

// ConfigNS 序列化
template <>
struct adl_serializer<ConfigNS>
{
    static void to_json(ordered_json &j, const ConfigNS &c)
    {
         adl_serializer<ConfigBase>::to_json(j,static_cast<const ConfigBase&>(c));
    }
    static void from_json(const ordered_json &j, ConfigNS &c)
    {
        adl_serializer<ConfigBase>::from_json(j, static_cast<ConfigBase&>(c));
    }
};

// 加载配置的模板函数
template <typename ConfigType>
ConfigType loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("配置文件打开失败: " + filename);
    nlohmann::ordered_json j;
    file >> j;
    return j.get<ConfigType>();
}

// 保存配置的模板函数
template <typename ConfigType>
void saveConfig(const ConfigType &config, const std::string &filename)
{
    nlohmann::ordered_json j = config;
    std::ofstream file(filename);
    file << j.dump(4);
}

} // namespace nlohmann