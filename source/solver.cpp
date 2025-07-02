#include "solver.h"
#include "constants.h"
#include "element.h"
#include "flux.h"
#include "macro.h"
#include <cstdint>
#include <fstream>
#include <iostream>

void Solver::Initialization()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            DataType dx = (config.x1 - config.x0) / config.n_ele;
            geom_pool[iele].x[isp] =
                config.x0 + iele * dx + isp * dx / (NSP - 1);
            if (geom_pool[iele].x[isp] < DataType(0.25))
            {
                elem_pool_old[iele].u_consrv[isp] = DataType(0.0);
            }
            else if (geom_pool[iele].x[isp] > DataType(0.75))
            {
                elem_pool_old[iele].u_consrv[isp] = DataType(0.0);
            }
            else
            {
                elem_pool_old[iele].u_consrv[isp] = DataType(1.0);
            }
        }

        // for (int isp = 0; isp < NSP; isp++)
        // {
        //     DataType dx = (config.x1 - config.x0) / config.n_ele;
        //     geom_pool[iele].x[isp] =
        //         config.x0 + iele * dx + isp * dx / (NSP - 1);

        //     elem_pool_old[iele].u_consrv[isp] =
        //         std::sin(2.0 * 3.1415926 * geom_pool[iele].x[isp]);
        // }

        computeElementGrad(iele);
    }
};
void Solver::computeElemRhs(Rhs *rhs_pool, Element *elem_pool, int iele)
{
    Element &element = elem_pool[iele];
    DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
    Element &element_l = elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
    Element &element_r = elem_pool[(iele + 1 + config.n_ele) % config.n_ele];

    DataType rhs_tmp[NSP]{};

    Flux *flux = new LinearAdvectionDiffusionFlux(config.a);
    DataType flux_tmp[NSP]{};

    for (int isp = 0; isp < NSP; isp++)
    {
        flux_tmp[isp] = flux->computeFlux(element.u_consrv[isp]);
    }

    DataType rhs_prediction[NSP]{};
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int jsp = 0; jsp < NSP; jsp++)
        {
            rhs_prediction[isp] +=
                getSMatrix<DataType, ORDER>()[jsp][isp] * flux_tmp[jsp];
        }

        rhs_tmp[isp] += rhs_prediction[isp];
    }

    // at  j - 1/2
    DataType rhs_common_flux_left = flux->computeRiemannFlux(
        element_l.u_consrv[ORDER], element.u_consrv[0]);
    // at j+1/2
    DataType rhs_common_flux_right = flux->computeRiemannFlux(
        element.u_consrv[ORDER], element_r.u_consrv[0]);
    rhs_tmp[0] += rhs_common_flux_left;
    rhs_tmp[ORDER] -= rhs_common_flux_right;

    for (int isp = 0; isp < NSP; isp++)
    {
        rhs_pool[iele].rhs[isp] = DataType(0.0);
        for (int jsp = 0; jsp < NSP; jsp++)
        {
            rhs_pool[iele].rhs[isp] +=
                DataType(2.0) / hj *
                invertMatrix<DataType, ORDER>(
                    getMMatrix<DataType, ORDER>())[isp][jsp] *
                rhs_tmp[jsp];
        }
    }
}
void Solver::computeRhs(Rhs *rhs_pool, Element *elem_pool)
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        // set rhs=0

        computeElemRhs(rhs_pool, elem_pool, iele);
    }
    return;
};

void Solver::timeRK1()
{

    computeRhs(rhs_pool_tmp, elem_pool_old);
    for (int iele = 0; iele < config.n_ele; iele++)
    {

        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_old[iele].u_consrv[isp] +=
                rhs_pool_tmp[iele].rhs[isp] * config.dt;
        }
    }
}

void Solver::timeRK2()
{

    computeRhs(rhs_pool_tmp, elem_pool_old);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_tmp[iele].u_consrv[isp] =
                elem_pool_old[iele].u_consrv[isp] +
                rhs_pool_tmp[iele].rhs[isp] * config.dt;
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {

        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_old[iele].u_consrv[isp] =
                DataType(0.5) * elem_pool_old[iele].u_consrv[isp] +
                DataType(0.5) * elem_pool_tmp[iele].u_consrv[isp] +
                DataType(0.5) * rhs_pool_tmp[iele].rhs[isp] * config.dt;
        }
    }
}

void Solver::timeRK3()
{
    computeRhs(rhs_pool_tmp, elem_pool_old);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_tmp[iele].u_consrv[isp] =
                elem_pool_old[iele].u_consrv[isp] +
                rhs_pool_tmp[iele].rhs[isp] * config.dt;
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_tmp[iele].u_consrv[isp] =
                DataType(0.75) * elem_pool_old[iele].u_consrv[isp] +
                DataType(0.25) * elem_pool_tmp[iele].u_consrv[isp] +
                DataType(0.25) * rhs_pool_tmp[iele].rhs[isp] * config.dt;
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            elem_pool_old[iele].u_consrv[isp] =
                DataType(1.0) / DataType(3.0) *
                    elem_pool_old[iele].u_consrv[isp] +
                DataType(2.0) / DataType(3.0) *
                    elem_pool_tmp[iele].u_consrv[isp] +
                DataType(2.0) / DataType(3.0) * rhs_pool_tmp[iele].rhs[isp] *
                    config.dt;
        }
    }
}

void Solver::computeElementGrad(int ielem)
{
    Element &element = elem_pool_old[ielem];

    for (int isp = 0; isp < NSP; isp++)
    {
        element.u_grad_consrv[isp] = DataType(0);
        for (int jsp = 0; jsp < NSP; jsp++)
        {
            element.u_grad_consrv[isp] +=
                getDMatrix<DataType, ORDER>()[isp][jsp] * element.u_consrv[jsp];
        }
    }
}

void Solver::Output(const std::string &filename)
{
    if (std::filesystem::exists(filename))
    {
        std::cerr << "Error: File " << filename << " already exists."
                  << std::endl;
        return;
    }
    std::ofstream ofile(filename);
    if (!ofile)
    {
        std::cerr << "Error: Failed to create file." << std::endl;
        return;
    }
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << geom_pool[iele].x[isp] << ",";
        }
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << elem_pool_old[iele].u_consrv[isp] << ",";
        }
        ofile << std::endl;
    }
    return;
};