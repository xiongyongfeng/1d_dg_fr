#include "solver.h"
#include "constants.h"
#include "element.h"
#include "macro.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <valarray>
#if defined(NS)
#include "ns.h"
#endif

#if defined(LAD)
#include "lad.h"
#endif

void Solver::Initialization()
{

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &elem = elem_pool_old[iele];
#ifdef LAD
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                DataType dx = (config.x1 - config.x0) / config.n_ele;
                geom_pool[iele].x[isp] =
                    config.x0 + iele * dx + isp * dx / (NSP - 1);
                if (geom_pool[iele].x[isp] < DataType(0.25))
                {
                    elem.u_consrv[isp][ivar] = DataType(10.0);
                }
                else if (geom_pool[iele].x[isp] > DataType(0.75))
                {
                    elem.u_consrv[isp][ivar] = DataType(10.0);
                }
                else
                {
                    elem.u_consrv[isp][ivar] = DataType(11.0);
                }
            }
        }
#endif

#ifdef NS
        DataType elem_consrv_left[NCONSRV];
        DataType elem_consrv_right[NCONSRV];
        DataType elem_primtv_left[NPRIMTV];
        DataType elem_primtv_right[NPRIMTV];
        elem_primtv_left[0] = 1.0;
        elem_primtv_left[1] = 0.0;
        elem_primtv_left[2] = 1.0;
        elem_primtv_left[3] = 1.0;
        elem_primtv_right[0] = 0.125;
        elem_primtv_right[1] = 0.0;
        elem_primtv_right[2] = 0.1;
        elem_primtv_right[3] = 0.1 / 0.125;
        Primtv2Consrv(elem_primtv_left, elem_consrv_left);
        Primtv2Consrv(elem_primtv_right, elem_consrv_right);
        for (int ivar = 0; ivar < NCONSRV; ivar++)
            printf("consrv_l[ivar] = %f, consr_r[ivar]=%f\n",
                   elem_consrv_left[ivar], elem_consrv_right[ivar]);
        for (int isp = 0; isp < NSP; isp++) /*for sod initialization*/
        {
            DataType dx = (config.x1 - config.x0) / config.n_ele;
            geom_pool[iele].x[isp] =
                config.x0 + iele * dx + isp * dx / (NSP - 1);
            if (geom_pool[iele].x[isp] < DataType(0.0))
            {
                elem.u_consrv[isp][0] = elem_consrv_left[0];
                elem.u_consrv[isp][1] = elem_consrv_left[1];
                elem.u_consrv[isp][2] = elem_consrv_left[2];
            }
            else
            {
                elem.u_consrv[isp][0] = elem_consrv_right[0];
                elem.u_consrv[isp][1] = elem_consrv_right[1];
                elem.u_consrv[isp][2] = elem_consrv_right[2];
            }
        }
#endif

        computeElementGrad(iele);
        ComputeElementAvg(iele);
    }
};
void Solver::computeElemRhsDG(Rhs *rhs_pool, Element *elem_pool, int iele)
{
    Element &element = elem_pool[iele];
    DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
    DataType local_det_jac = hj / DataType(2.0);
    Element &element_l = elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
    Element &element_r = elem_pool[(iele + 1 + config.n_ele) % config.n_ele];

    DataType rhs_tmp[NSP][NCONSRV]{};

    DataType flux_tmp[NSP][NCONSRV]{};

    for (int isp = 0; isp < NSP; isp++)
    {
        computeFlux(element.u_consrv[isp], flux_tmp[isp], config.a);
    }
#if 0
    if (iele == 99)
    {

        for (int isp = 0; isp < NSP; isp++)
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                printf("DG: ielem =%d, u_consrv[%d][%d]=%f\n", iele, isp,
                ivar,
                       element.u_consrv[isp][ivar]);
                printf("DG: ielem =%d, flux_tmp[%d][%d]=%f\n", iele, isp,
                ivar,
                       flux_tmp[isp][ivar]);
            }
    }
#endif

    DataType rhs_prediction[NSP][NCONSRV]{};
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_prediction[isp][ivar] +=
                    getSMatrix<DataType, ORDER>()[jsp][isp] *
                    flux_tmp[jsp][ivar];
            }
            rhs_tmp[isp][ivar] += rhs_prediction[isp][ivar];
        }
    }

    // at  j - 1/2
    DataType rhs_common_flux_left[NCONSRV];
    computeRiemannFlux(element_l.u_consrv[ORDER], element.u_consrv[0],
                       rhs_common_flux_left, config.a);

#ifdef NS
    if (iele == 0)
    {
        computeRiemannFlux(element.u_consrv[0], element.u_consrv[0],
                           rhs_common_flux_left, config.a);
    }
#endif
    // at j+1/2
    DataType rhs_common_flux_right[NCONSRV];
    computeRiemannFlux(element.u_consrv[ORDER], element_r.u_consrv[0],
                       rhs_common_flux_right, config.a);

#ifdef NS
    if (iele == config.n_ele - 1)
    {
        computeRiemannFlux(element.u_consrv[ORDER], element.u_consrv[ORDER],
                           rhs_common_flux_right, config.a);
    }
#endif

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        rhs_tmp[0][ivar] += rhs_common_flux_left[ivar];
        rhs_tmp[ORDER][ivar] -= rhs_common_flux_right[ivar];
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_pool[iele].rhs[isp][ivar] = DataType(0.0);
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_pool[iele].rhs[isp][ivar] +=
                    DataType(2.0) / hj *
                    invertMatrix<DataType, ORDER>(
                        getMMatrix<DataType, ORDER>())[isp][jsp] *
                    rhs_tmp[jsp][ivar];
            }
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

    DataType flux_tmp[NSP][NCONSRV]{};

    for (int isp = 0; isp < NSP; isp++)
    {
        computeFlux(element.u_consrv[isp], flux_tmp[isp], config.a);
    }
#if 0
    if (iele == 0)
    {
        for (int isp = 0; isp < NSP; isp++)
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                printf("FR: ielem =%d, u_consrv[%d][%d]=%f\n", iele, isp, ivar,
                       element.u_consrv[isp][ivar]);
                printf("FR: ielem =%d, flux_tmp[%d][%d]=%f\n", iele, isp, ivar,
                       flux_tmp[isp][ivar]);
            }
    }
#endif

    DataType rhs_prediction[NSP][NCONSRV]{};

    // // chain rule
    // for (int isp = 0; isp < NSP; isp++)
    // {
    //     for (int ivar = 0; ivar < NCONSRV; ivar++)
    //     {
    //         element.u_grad_consrv[isp][ivar] = 0.0;
    //         for (int jsp = 0; jsp < NSP; jsp++)
    //         {
    //             element.u_grad_consrv[isp][ivar] +=
    //                 getDMatrix<DataType, ORDER>()[isp][jsp] *
    //                 element.u_consrv[jsp][ivar] / local_det_jac;
    //         }
    //     }
    // }

    // for (int isp = 0; isp < NSP; isp++)
    // {
    //     DataType rho = element.u_consrv[isp][0];
    //     DataType rho_u = element.u_consrv[isp][1];
    //     DataType rho_E = element.u_consrv[isp][2];
    //     DataType d_rho = element.u_grad_consrv[isp][0];
    //     DataType d_rhou = element.u_grad_consrv[isp][1];
    //     DataType d_rhoE = element.u_grad_consrv[isp][2];
    //     DataType u = rho_u / rho;

    //     rhs_prediction[isp][0] -= d_rhou;
    //     rhs_prediction[isp][1] -= (-0.5 * u * u * (3.0 - GAMMA)) * d_rho +
    //                               u * (3.0 - GAMMA) * d_rhou +
    //                               (GAMMA - 1.0) * d_rhoE;
    //     rhs_prediction[isp][2] -=
    //         (-rho_u * rho_E / rho / rho * GAMMA + (GAMMA - 1.0) * u * u * u)
    //         *
    //             d_rho +
    //         (rho_E / rho * GAMMA - 1.5 * (GAMMA - 1.0) * u * u) * d_rhou +
    //         GAMMA * u * d_rhoE;
    // }
    // // chain rule

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_prediction[isp][ivar] -=
                    getDMatrix<DataType, ORDER>()[isp][jsp] *
                    flux_tmp[jsp][ivar] / local_det_jac;
            }
        }
    }

#if 0
    if (iele == 0)
    {
        for (int isp = 0; isp < NSP; isp++)
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                printf("FR: ielem =%d, rhs_prediction[%d][%d]=%f\n", iele, isp,
                       ivar, rhs_prediction[isp][ivar]);
            }
    }
#endif

    // at  j - 1/2
    DataType common_flux_left[NCONSRV];
    computeRiemannFlux(element_l.u_consrv[ORDER], element.u_consrv[0],
                       common_flux_left, config.a);
#if 0
    if (iele == 0)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            printf("FR: ielem =%d, common_flux_left[%d]=%f, ul=%f,ur=%f, "
                   "config.a = %f\n",
                   iele, ivar, common_flux_left[ivar],
                   element_l.u_consrv[ORDER][0], element.u_consrv[0][0],
                   config.a);
        }
    }
#endif
#ifdef NS
    if (iele == 0)
    {
        computeRiemannFlux(element.u_consrv[0], element.u_consrv[0],
                           common_flux_left, config.a);
    }
#endif
    // at j+1/2
    DataType common_flux_right[NCONSRV];
    computeRiemannFlux(element.u_consrv[ORDER], element_r.u_consrv[0],
                       common_flux_right, config.a);
#ifdef NS
    if (iele == config.n_ele - 1)
    {
        computeRiemannFlux(element.u_consrv[ORDER], element.u_consrv[ORDER],
                           common_flux_right, config.a);
    }
#endif

    DataType rhs_correction[NSP][NCONSRV]{};

    DataType flux_tmp2[NSP][NCONSRV]{};
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {

        flux_tmp2[0][ivar] =
            -1.0 * (common_flux_left[ivar] - flux_tmp[0][ivar]);
        flux_tmp2[ORDER][ivar] =
            1.0 * (common_flux_right[ivar] - flux_tmp[ORDER][ivar]);
    }
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_correction[isp][ivar] -=
                    invertMatrix<DataType, ORDER>(
                        getMMatrix<DataType, ORDER>())[isp][jsp] *
                    flux_tmp2[jsp][ivar] / local_det_jac;
            }
        }
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_pool[iele].rhs[isp][ivar] = DataType(0.0);
            rhs_pool[iele].rhs[isp][ivar] +=
                rhs_prediction[isp][ivar] + rhs_correction[isp][ivar];
        }
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
#if 0
            if (iele >= 98 && iele <= 100)
            {
                for (int isp = 0; isp < NSP; isp++)
                    for (int ivar = 0; ivar < NCONSRV; ivar++)
                        printf("DG: rhs_pool[%d].rhs[%d][%d]=%f\n", iele,
                        isp,
                               ivar, rhs_pool[iele].rhs[isp][ivar]);
            }
#endif
        }
        else if (config.dg_fr_type == 1)
        {
            computeElemRhsFR(rhs_pool, elem_pool, iele);
#if 0
            if (iele == 0)
            {
                for (int isp = 0; isp < NSP; isp++)
                    for (int ivar = 0; ivar < NCONSRV; ivar++)
                        printf("FR: rhs_pool[%d].rhs[%d][%d]=%f\n", iele, isp,
                               ivar, rhs_pool[iele].rhs[isp][ivar]);
            }
#endif
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
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[iele].u_consrv[isp][ivar] +=
                    rhs_pool_tmp[iele].rhs[isp][ivar] * config.dt;
            }
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
    DataType c1[config.n_ele][NCONSRV];
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &element = elem_pool_old[iele];
        DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
        Element &element_l =
            elem_pool_old[(iele - 1 + config.n_ele) % config.n_ele];
        Element &element_r =
            elem_pool_old[(iele + 1 + config.n_ele) % config.n_ele];

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            std::pair<DataType, int> c1p =
                Minmod(element.u_consrv[1][ivar] - element.u_avg[ivar],
                       element_r.u_avg[ivar] - element.u_avg[ivar],
                       element.u_avg[ivar] - element_l.u_avg[ivar]);
            std::pair<DataType, int> c1m =
                Minmod(-element.u_consrv[0][ivar] + element.u_avg[ivar],
                       element_r.u_avg[ivar] - element.u_avg[ivar],
                       element.u_avg[ivar] - element_l.u_avg[ivar]);
            c1[iele][ivar] = DataType(0.5) * (c1p.first + c1m.first);
            element.islimited[ivar] = c1p.second + c1m.second;
        }
    }

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &element = elem_pool_old[iele];
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                if (element.islimited[ivar] > 0)
                {

                    element.u_consrv[isp][ivar] =
                        element.u_avg[ivar] +
                        c1[iele][ivar] * getLGLPoints<DataType, ORDER>()[isp];
                }
            }
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
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {

                elem_pool_tmp[iele].u_consrv[isp][ivar] =
                    elem_pool_old[iele].u_consrv[isp][ivar] +
                    rhs_pool_tmp[iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[iele].u_consrv[isp][ivar] =
                    DataType(0.5) * elem_pool_old[iele].u_consrv[isp][ivar] +
                    DataType(0.5) * elem_pool_tmp[iele].u_consrv[isp][ivar] +
                    DataType(0.5) * rhs_pool_tmp[iele].rhs[isp][ivar] *
                        config.dt;
            }
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
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[iele].u_consrv[isp][ivar] =
                    elem_pool_old[iele].u_consrv[isp][ivar] +
                    rhs_pool_tmp[iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[iele].u_consrv[isp][ivar] =
                    DataType(0.75) * elem_pool_old[iele].u_consrv[isp][ivar] +
                    DataType(0.25) * elem_pool_tmp[iele].u_consrv[isp][ivar] +
                    DataType(0.25) * rhs_pool_tmp[iele].rhs[isp][ivar] *
                        config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp, elem_pool_tmp);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[iele].u_consrv[isp][ivar] =
                    DataType(1.0) / DataType(3.0) *
                        elem_pool_old[iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        elem_pool_tmp[iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        rhs_pool_tmp[iele].rhs[isp][ivar] * config.dt;
            }
        }
    }
}

void Solver::computeElementGrad(int ielem)
{
    Element &element = elem_pool_old[ielem];

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            element.u_grad_consrv[isp][ivar] = DataType(0);
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                element.u_grad_consrv[isp][ivar] +=
                    getDMatrix<DataType, ORDER>()[isp][jsp] *
                    element.u_consrv[jsp][ivar];
            }
        }
    }
}
void Solver::ComputeElementAvg(int ielem)
{
    Element &element = elem_pool_old[ielem];
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        element.u_avg[ivar] = DataType(0);
        for (int isp = 0; isp < NSP; isp++)
        {
            element.u_avg[ivar] += getLGLWeights<DataType, ORDER>()[isp] *
                                   element.u_consrv[isp][ivar];
        }
        element.u_avg[ivar] /= DataType(2.0);
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
        const Element &elem = elem_pool_old[iele];
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << geom_pool[iele].x[isp] << ",";
        }

        DataType elem_primtv[NSP][NPRIMTV];

        for (int isp = 0; isp < NSP; isp++)
        {

            DataType primtv[NPRIMTV];
            Consrv2Primtv(elem_pool_old[iele].u_consrv[isp], primtv);
            for (int ivar = 0; ivar < NPRIMTV; ivar++)
            {
                elem_primtv[isp][ivar] = primtv[ivar];
            }
        }
        for (int ivar = 0; ivar < NPRIMTV; ivar++)
        {
            for (int isp = 0; isp < NSP; isp++)
            {
                ofile << elem_primtv[isp][ivar] << ",";
            }
        }

        ofile << std::endl;
    }
    return;
};