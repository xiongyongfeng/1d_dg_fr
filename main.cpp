#include "base/constants.h"
#include "base/element.h"
#include "base/macro.h"
#include "base/parser.h"
#include "base/solver_config.h"
#include "solver/solver.h"
#include <cstdlib>
#include <iostream>
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
    SolverConfig solver_config = nlohmann::loadConfig(argv[1]);
    Element *elem_pool = new Element[solver_config.n_ele];

    Solver solver(solver_config, elem_pool);

    solver.Initialization();
    solver.Compute();

    std::string filename("result.csv");
    solver.Output(filename);

    ///////////////////////////////
    const auto &m = getMMatrix<DataType, ORDER>();
    for (int i = 0; i < ORDER + 1; i++)
    {
        for (int j = 0; j < ORDER + 1; j++)
        {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }

    //////////////////////////////

    delete[] elem_pool;
    std::cout << "Finish Computation!" << std::endl;
    return 0;
}