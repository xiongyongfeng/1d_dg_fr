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
        } // 若目录已存在则无需操作
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
    Solver solver(config, config.n_ele);

    solver.Initialization();

    DataType current_time = 0;
    while (current_time <= config.total_time)
    {
        if (static_cast<int>(std::round(current_time / config.dt)) %
                static_cast<int>(
                    std::round(config.output_time_step / config.dt)) ==
            0)
        {
            ensurePathExists(config.output_dir);
            std::string filename = config.output_dir + "/result_" +
                                   std::to_string(current_time) + ".csv";
            std::cout << "output file: " << filename << std::endl;
            solver.Output(filename);
        }

        solver.timeRK3();

        current_time += config.dt;
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