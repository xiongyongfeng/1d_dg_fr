#include "solver.h"
#include "constants.h"
#include "element.h"
#include "flux.h"
#include "macro.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <valarray>

void Solver::Initialization()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &elem = elem_pool_old[iele];
        for (int isp = 0; isp < NSP; isp++)
        {
            DataType dx = (config.x1 - config.x0) / config.n_ele;
            geom_pool[iele].x[isp] =
                config.x0 + iele * dx + isp * dx / (NSP - 1);
            if (geom_pool[iele].x[isp] < DataType(0.25))
            {
                elem.u_consrv[isp] = DataType(10.0);
            }
            else if (geom_pool[iele].x[isp] > DataType(0.75))
            {
                elem.u_consrv[isp] = DataType(10.0);
            }
            else
            {
                elem.u_consrv[isp] = DataType(11.0);
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
        ComputeElementAvg(iele);
    }
};
void Solver::computeElemRhsDG(Rhs *rhs_pool, Element *elem_pool, int iele)
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

void Solver::computeElemRhsFR(Rhs *rhs_pool, Element *elem_pool, int iele)
{
    Element &element = elem_pool[iele];
    DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
    DataType local_det_jac = hj / DataType(2.0);
    Element &element_l = elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
    Element &element_r = elem_pool[(iele + 1 + config.n_ele) % config.n_ele];

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
            rhs_prediction[isp] -= getDMatrix<DataType, ORDER>()[isp][jsp] *
                                   flux_tmp[jsp] / local_det_jac;
        }
    }

    // at  j - 1/2
    DataType common_flux_left = flux->computeRiemannFlux(
        element_l.u_consrv[ORDER], element.u_consrv[0]);
    // at j+1/2
    DataType common_flux_right = flux->computeRiemannFlux(
        element.u_consrv[ORDER], element_r.u_consrv[0]);

    DataType rhs_correction[NSP]{};

    DataType flux_tmp2[NSP]{};
    flux_tmp2[0] = -1.0 * (common_flux_left - flux_tmp[0]);
    flux_tmp2[ORDER] = 1.0 * (common_flux_right - flux_tmp[ORDER]);
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int jsp = 0; jsp < NSP; jsp++)
        {
            rhs_correction[isp] -=
                invertMatrix<DataType, ORDER>(
                    getMMatrix<DataType, ORDER>())[isp][jsp] *
                flux_tmp2[jsp] / local_det_jac;
        }
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        rhs_pool[iele].rhs[isp] = DataType(0.0);
        rhs_pool[iele].rhs[isp] += rhs_prediction[isp] + rhs_correction[isp];
    }
}
void Solver::computeRhs(Rhs *rhs_pool, Element *elem_pool)
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        // set rhs=0
        if (config.dg_fr_type == 0)
        {
            computeElemRhsDG(rhs_pool, elem_pool, iele);
        }
        else if (config.dg_fr_type == 1)
        {
            computeElemRhsFR(rhs_pool, elem_pool, iele);
        }
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

void Solver::Post()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        ComputeElementAvg(iele);
        computeElementGrad(iele);
    }
    return;
}

std::pair<DataType, int> Solver::Minmod(DataType a, DataType b, DataType c)
{
    // 检查所有参数是否同号
    if (a > -1e-6 && a < 1e-6)
        return {DataType(a), 0};
    ;

    const bool all_nonneg = (a >= 0) && (b >= 0) && (c >= 0);
    const bool all_nonpos = (a <= 0) && (b <= 0) && (c <= 0);

    // 符号不一致时激活 Limiter
    if (!(all_nonneg || all_nonpos))
    {
        return {DataType(0), 1}; // 返回 0 且标记 Limiter 激活
    }

    // 同号时选择绝对值最小的参数
    const DataType min_abs = std::min({std::abs(a), std::abs(b), std::abs(c)});
    const DataType result = (a > 0) ? min_abs : -min_abs;

    // 检查是否实际应用了 Limiter（是否返回了非原始输入值）
    const bool limiter_activated = (std::abs(result - a) > 1e-5 ? 1 : 0);
    return {result, limiter_activated};
}

void Solver::TvdLimiter()
{
    Post();

    bool islimited[config.n_ele];
    DataType c1[config.n_ele];
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &element = elem_pool_old[iele];
        DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
        Element &element_l =
            elem_pool_old[(iele - 1 + config.n_ele) % config.n_ele];
        Element &element_r =
            elem_pool_old[(iele + 1 + config.n_ele) % config.n_ele];
        std::pair<DataType, int> c1p = Minmod(
            element.u_consrv[1] - element.u_avg,
            element_r.u_avg - element.u_avg, element.u_avg - element_l.u_avg);
        std::pair<DataType, int> c1m = Minmod(
            -element.u_consrv[0] + element.u_avg,
            element_r.u_avg - element.u_avg, element.u_avg - element_l.u_avg);
        // std::cout << "iele = " << iele << ", islimited = " << c1p.second << "
        // "
        //           << c1m.second << " " << c1p.first << " " << c1m.first
        //           << std::endl;
        // printf("a=%f,b=%f,c=%f\n", element.u_consrv[1] - element.u_avg,
        //        element_r.u_avg - element.u_avg,
        //        element.u_avg - element_l.u_avg);
        c1[iele] = DataType(0.5) * (c1p.first + c1m.first);
        element.islimited = c1p.second + c1m.second;
    }

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &element = elem_pool_old[iele];
        if (element.islimited > 0)
            for (int isp = 0; isp < NSP; isp++)
            {
                element.u_consrv[isp] =
                    element.u_avg +
                    c1[iele] * getLGLPoints<DataType, ORDER>()[isp];
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
void Solver::ComputeElementAvg(int ielem)
{
    Element &element = elem_pool_old[ielem];

    element.u_avg = DataType(0);
    for (int isp = 0; isp < NSP; isp++)
    {
        element.u_avg +=
            getLGLWeights<DataType, ORDER>()[isp] * element.u_consrv[isp];
    }
    element.u_avg /= DataType(2.0);
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
        ofile << elem_pool_old[iele].islimited << ",";
        ofile << std::endl;
    }
    return;
};