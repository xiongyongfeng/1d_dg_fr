#include "solver.h"
#include <cstdint>
#include <fstream>
#include <iostream>

void Solver::Initialization()
{
    for (int iele = 0; iele < solver_config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            DataType dx =
                (solver_config.x1 - solver_config.x0) / solver_config.n_ele;
            elem_pool[iele].x[isp] =
                solver_config.x0 + iele * dx + isp * dx / (NSP - 1);
            if (elem_pool[iele].x[isp] < DataType(0.25))
            {
                elem_pool[iele].u_consrv[isp] = DataType(0.0);
            }
            else if (elem_pool[iele].x[isp] > DataType(0.75))
            {
                elem_pool[iele].u_consrv[isp] = DataType(0.0);
            }
            else
            {
                elem_pool[iele].u_consrv[isp] = DataType(1.0);
            }
        }
    }
};
void Solver::Compute(){};
void Solver::Output(const std::string &filename)
{

    std::ofstream ofile(filename);
    for (int iele = 0; iele < solver_config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << elem_pool[iele].x[isp] << ",";
        }
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << elem_pool[iele].u_consrv[isp] << ",";
        }
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << elem_pool[iele].u_grad_consrv[isp] << ",";
        }
        ofile << std::endl;
    }
};