#include "config.h"
#include "constants.h"
#include "element.h"
#include "macro.h"
#include "parser.h"
#include "solver.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
namespace fs = std::filesystem;

void ensurePathExists(const fs::path &output_dir)
{
    try
    {
        if (!fs::exists(output_dir))
        {
            bool created =
                fs::create_directories(output_dir); // 递归创建所有缺失目录
            if (created)
            {
                std::cout << "已创建目录: " << output_dir << std::endl;
            }
        }
        else
        {

            std::cerr << "\nError: Directory " << output_dir
                      << " already exists.\n";
            exit(EXIT_FAILURE);
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "目录创建失败: " << e.what() << std::endl;
        // 可在此处终止程序或抛出异常
    }
}
int main(int argc, char **argv)
{
    std::cout << "running command: ";
    for (int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;
    if (argc < 1)
    {
        std::cout << "usage: exe *.json" << std::endl;
        exit(-1);
    }
    Config config = nlohmann::loadConfig(argv[1]);
    nlohmann::json j = config; // 自动序列化
    std::cout << j.dump(4);    // 缩进4格输出，便于阅读
    ensurePathExists(config.output_dir);
    ensurePathExists(config.output_dir + "_avg");
    Solver solver(config, config.n_ele);

    solver.Initialization();
    DataType current_time = 0;

    if (static_cast<int>(std::round(current_time / config.dt)) %
            static_cast<int>(std::round(config.output_time_step / config.dt)) ==
        0)
    {
        std::string filename =
            config.output_dir + "/init" + std::to_string(current_time) + ".csv";
        std::cout << "output file: " << filename << std::endl;
        solver.Output(filename);

        std::string filename_avg = config.output_dir + "_avg/init" +
                                   std::to_string(current_time) + ".csv";
        std::cout << "output file: " << filename_avg << std::endl;
        solver.OutputAvg(filename_avg);
    }

    while (current_time <= config.total_time)
    {

        if (config.time_scheme_type == 0)
        {
            // solver.timeRK3();
            solver.timeRK1();
        }
        if (config.time_scheme_type == 1)
        {
            solver.timeNewExplicitSchemeKN();
        }
        current_time += config.dt;

        if (static_cast<int>(std::round(current_time / config.dt)) %
                static_cast<int>(
                    std::round(config.output_time_step / config.dt)) ==
            0)
        {
            std::string filename = config.output_dir + "/result_before" +
                                   std::to_string(current_time) + ".csv";
            std::cout << "output file: " << filename << std::endl;
            solver.Output(filename);

            std::string filename_avg = config.output_dir +
                                       "_avg/result_before" +
                                       std::to_string(current_time) + ".csv";
            std::cout << "output file: " << filename_avg << std::endl;
            solver.OutputAvg(filename_avg);
        }

        if (config.limiter_type == 1)
        {
            solver.TvdLimiter();
        }

        if (static_cast<int>(std::round(current_time / config.dt)) %
                static_cast<int>(
                    std::round(config.output_time_step / config.dt)) ==
            0)
        {
            std::string filename = config.output_dir + "/result_after" +
                                   std::to_string(current_time) + ".csv";
            std::cout << "output file: " << filename << std::endl;
            solver.Output(filename);

            std::string filename_avg = config.output_dir + "_avg/result_after" +
                                       std::to_string(current_time) + ".csv";
            std::cout << "output file: " << filename_avg << std::endl;
            solver.OutputAvg(filename_avg);
        }
    }

    ///////////////////////////////
    // const auto &m = getMMatrix<DataType, ORDER>();
    // for (int i = 0; i < ORDER + 1; i++)
    // {
    //     for (int j = 0; j < ORDER + 1; j++)
    //     {
    //         std::cout << m[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    //////////////////////////////
    std::cout << "Finish Computation!" << std::endl;
    return 0;
}