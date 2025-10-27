#include "solver.h"
#include "constants.h"
#include "element.h"
#include "initializer.h"
#include "macro.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <valarray>
#if defined(NS)
#include "ns.h"
#endif

#if defined(LAD)
#include "lad.h"
#endif

#if defined(BURGERS)
#include "burgers.h"
#endif

void Solver::Initialization()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Geom &geom = geom_pool[iele];
        for (int isp = 0; isp < NSP; isp++) /*for sod initialization*/
        {
            DataType dx = (config.x1 - config.x0) / config.n_ele;
            geom.x[isp] = config.x0 + iele * dx + isp * dx / (NSP - 1);
        }
        geom.dx = geom.x[ORDER] - geom.x[0];
        geom.local_det_jac = geom.dx / 2.0;
    }

    for (int itp = 0; itp < NTP; itp++)
    {
        DataType t =
            (getLGLPoints<DataType, TORDER>()[itp] + 1.0) / 2.0 * config.dt;
        for (int iele = 0; iele < config.n_ele; iele++)
        {
            Element &elem = elem_pool_old[itp][iele];

#ifdef LAD
            for (int isp = 0; isp < NSP; isp++)
            {
                DataType x = geom_pool[iele].x[isp];
                for (int ivar = 0; ivar < NCONSRV; ivar++)
                {
                    elem.u_consrv[isp][ivar] = exact_lad(x, 0)[ivar];
                }
            }
#endif

            computeElementGrad(iele);
            ComputeElementAvg(iele);
        }
    }

    for (int itp = 0; itp < NTP; itp++) // init  the first step with RK4
    {
        DataType t =
            (getLGLPoints<DataType, TORDER>()[itp] + 1.0) / 2.0 * config.dt;
        for (int iele = 0; iele < config.n_ele; iele++)
        {
            Element &elem = elem_pool_old[itp][iele];

#ifdef BURGERS
            if (itp == 0)
            {
                // if(1){
                for (int isp = 0; isp < NSP; isp++)
                {
                    for (int ivar = 0; ivar < NCONSRV; ivar++)
                    {
                        DataType x = geom_pool[iele].x[isp];
                        elem.u_consrv[isp][ivar] = u0_burgers(x)[ivar];
                    }
                }
            }
            else
            {
                init_RK3(itp);
            }

#endif

#ifdef NS
            if (itp == 0)
            {
                for (int isp = 0; isp < NSP; isp++) /*for sod initialization*/
                {
                    DataType x = geom_pool[iele].x[isp];
                    for (int ivar = 0; ivar < NCONSRV; ivar++)
                    {
                        elem.u_consrv[isp][ivar] = u0_sod(x)[ivar];
                    }
                }
            }
            else
            {
                init_RK3(itp);
            }
#endif

            computeElementGrad(iele);
            ComputeElementAvg(iele);
        }
    }
};
void Solver::computeElemRhsDG(Rhs *rhs_pool, const Element *elem_pool, int iele)
{
    const Element &element = elem_pool[iele];
    DataType local_det_jac = geom_pool[iele].local_det_jac;
#ifdef NS
    const Element &element_l = elem_pool[iele == 0 ? iele : iele - 1];
#else
    const Element &element_l =
        elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
#endif
    DataType local_det_jac_L =
        geom_pool[(iele - 1 + config.n_ele) % config.n_ele].local_det_jac;
#ifdef NS
    const Element &element_r =
        elem_pool[iele == config.n_ele - 1 ? iele : iele + 1];
#else
    const Element &element_r =
        elem_pool[(iele + 1 + config.n_ele) % config.n_ele];
#endif
    DataType local_det_jac_R =
        geom_pool[(iele + 1 + config.n_ele) % config.n_ele].local_det_jac;

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
                    Constant_s.getSMatrix()[jsp][isp] * flux_tmp[jsp][ivar];
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

#ifdef LAD
    // add diffusion term
    DataType visflux_tmp[NSP][NCONSRV]{};
    DataType rhs_common_visflux_left[NCONSRV];
    DataType rhs_common_visflux_right[NCONSRV];
    DataType globalLift[NSP * NCONSRV] = {0.0};
    DataType globalLift_L[NSP * NCONSRV];
    DataType globalLift_R[NSP * NCONSRV];

    computeBR2Flux(element_l.u_consrv[ORDER], element_l.u_grad_consrv[ORDER],
                   element.u_consrv[0], element.u_grad_consrv[0],
                   local_det_jac_L, local_det_jac, rhs_common_visflux_left,
                   globalLift_L, globalLift, Constant_s, config.nu);
    computeBR2Flux(element.u_consrv[ORDER], element.u_grad_consrv[ORDER],
                   element_r.u_consrv[0], element_r.u_grad_consrv[0],
                   local_det_jac, local_det_jac_R, rhs_common_visflux_right,
                   globalLift, globalLift_R, Constant_s, config.nu);

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        rhs_tmp[0][ivar] -= rhs_common_visflux_left[ivar];
        rhs_tmp[ORDER][ivar] += rhs_common_visflux_right[ivar];
    }

    // add global lift's contribution on grad
    DataType grad_u_consrv_[NSP][NCONSRV];
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            grad_u_consrv_[isp][ivar] = element.u_grad_consrv[isp][ivar] -
                                        globalLift[isp * NCONSRV + ivar];
        }
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        computeVisFlux(element.u_consrv[isp], grad_u_consrv_[isp],
                       visflux_tmp[isp], config.nu);
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_tmp[isp][ivar] -=
                    Constant_s.getSMatrix()[jsp][isp] * visflux_tmp[jsp][ivar];
            }
        }
    }
#endif

#ifdef NS
    // add diffusion term
    DataType visflux_tmp[NSP][NCONSRV]{};
    DataType rhs_common_visflux_left[NCONSRV];
    DataType rhs_common_visflux_right[NCONSRV];
    DataType globalLift[NSP * NCONSRV] = {0.0};
    DataType globalLift_L[NSP * NCONSRV];
    DataType globalLift_R[NSP * NCONSRV];

    computeBR2Flux(element_l.u_consrv[ORDER], element_l.u_grad_consrv[ORDER],
                   element.u_consrv[0], element.u_grad_consrv[0],
                   local_det_jac_L, local_det_jac, rhs_common_visflux_left,
                   globalLift_L, globalLift, Constant_s, config.vis);
    computeBR2Flux(element.u_consrv[ORDER], element.u_grad_consrv[ORDER],
                   element_r.u_consrv[0], element_r.u_grad_consrv[0],
                   local_det_jac, local_det_jac_R, rhs_common_visflux_right,
                   globalLift, globalLift_R, Constant_s, config.vis);

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        rhs_tmp[0][ivar] -= rhs_common_visflux_left[ivar];
        rhs_tmp[ORDER][ivar] += rhs_common_visflux_right[ivar];
    }

    // add global lift's contribution on grad
    DataType grad_u_consrv_[NSP][NCONSRV];
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            grad_u_consrv_[isp][ivar] = element.u_grad_consrv[isp][ivar] -
                                        globalLift[isp * NCONSRV + ivar];
        }
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        computeVisFlux(element.u_consrv[isp], grad_u_consrv_[isp],
                       visflux_tmp[isp], config.vis);
    }

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_tmp[isp][ivar] -=
                    Constant_s.getSMatrix()[jsp][isp] * visflux_tmp[jsp][ivar];
            }
        }
    }
#endif

    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_pool[iele].rhs[isp][ivar] = DataType(0.0);
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_pool[iele].rhs[isp][ivar] +=
                    invertMatrix<DataType, ORDER>(
                        Constant_s.getMMatrix())[isp][jsp] *
                    rhs_tmp[jsp][ivar] / local_det_jac;
            }
        }
    }
}

void Solver::compPredictionLP(const DataType (&flux)[NSP][NCONSRV],
                              const DataType &local_det_jac,
                              DataType (&rhs_predict)[NSP][NCONSRV])
{
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_predict[isp][ivar] -= Constant_s.getDMatrix()[isp][jsp] *
                                          flux[jsp][ivar] / local_det_jac;
            }
        }
    }
}

void Solver::compPredictionEntropy(const DataType (&flux)[NSP][NCONSRV],
                                   const DataType (&consrv)[NSP][NCONSRV],
                                   const DataType &local_det_jac,
                                   DataType (&rhs_predict)[NSP][NCONSRV])
{
#ifdef LAD
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_predict[isp][ivar] -=
                    2.0 * Constant_s.getDMatrix()[isp][jsp] * 0.5 *
                    (flux[jsp][ivar] + flux[isp][ivar]) / local_det_jac;
            }
        }
    }
#endif

#ifdef BURGERS
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_predict[isp][ivar] -=
                    2.0 * Constant_s.getDMatrix()[isp][jsp] * 1.0 / 6.0 *
                    (consrv[isp][ivar] * consrv[isp][ivar] +
                     consrv[jsp][ivar] * consrv[isp][ivar] +
                     consrv[jsp][ivar] * consrv[jsp][ivar]) /
                    local_det_jac;
            }
        }
    }
#endif

#ifdef NS

    DataType primtv[NSP][NPRIMTV];
    for (int isp = 0; isp < NSP; isp++)
    {
        Consrv2Primtv(consrv[isp], primtv[isp]);
    }
    // Chandrashekar
    // for (int isp = 0; isp < NSP; isp++)
    // {
    //     // fs_0

    //     for (int jsp = 0; jsp < NSP; jsp++)
    //     {
    //         DataType rho_bar_log;
    //         DataType rho_R = primtv[isp][0];
    //         DataType rho_L = primtv[jsp][0];
    //         if (std::abs(rho_R - rho_L) / (std::abs(rho_L) + 1e-12) < 1e-6)
    //         {
    //             rho_bar_log = rho_L;
    //         }
    //         else
    //         {
    //             rho_bar_log =
    //                 (rho_R - rho_L) / (std::log(rho_R) - std::log(rho_L));
    //         }
    //         DataType u_R = primtv[isp][1];
    //         DataType u_L = primtv[jsp][1];
    //         DataType u_bar = 0.5 * (u_R + u_L);
    //         DataType fs_0 = rho_bar_log * u_bar;
    //         rhs_predict[isp][0] -= 2.0 *
    //                                getDMatrix<DataType, ORDER>()[isp][jsp] *
    //                                fs_0 / local_det_jac;

    //         DataType rho_bar = 0.5 * (rho_R + rho_L);

    //         DataType p_R = primtv[isp][2];
    //         DataType p_L = primtv[jsp][2];
    //         DataType beta_R = rho_R / p_R / 2.0;
    //         DataType beta_L = rho_L / p_L / 2.0;
    //         DataType beta_bar = 0.5 * (beta_R + beta_L);
    //         DataType fs_1 = rho_bar / 2.0 / beta_bar + u_bar * fs_0;

    //         rhs_predict[isp][1] -= 2.0 *
    //                                getDMatrix<DataType, ORDER>()[isp][jsp] *
    //                                fs_1 / local_det_jac;

    //         DataType beta_bar_log;
    //         if (std::abs(beta_R - beta_L) / (std::abs(beta_L) + 1e-12) <
    //         1e-6)
    //         {
    //             beta_bar_log = beta_L;
    //         }
    //         else
    //         {
    //             beta_bar_log =
    //                 (beta_R - beta_L) / (std::log(beta_R) -
    //                 std::log(beta_L));
    //         }
    //         DataType u_square_bar = 0.5 * (u_R * u_R + u_L * u_L);
    //         DataType fs_2 = (1.0 / (2.0 * (GAMMA - 1.0) * beta_bar_log) -
    //                          0.5 * u_square_bar) *
    //                             fs_0 +
    //                         u_bar * fs_1;
    //         rhs_predict[isp][2] -= 2.0 *
    //                                getDMatrix<DataType, ORDER>()[isp][jsp] *
    //                                fs_2 / local_det_jac;
    //     }
    // }

    // Ismail & Roe
    for (int isp = 0; isp < NSP; isp++)
    {
        // fs_0
        for (int jsp = 0; jsp < NSP; jsp++)
        {
            DataType rho_R = primtv[isp][0];
            DataType rho_L = primtv[jsp][0];
            DataType p_R = primtv[isp][2];
            DataType p_L = primtv[jsp][2];
            DataType u_R = primtv[isp][1];
            DataType u_L = primtv[jsp][1];
            DataType z1_L = std::sqrt(rho_L / p_L);
            DataType z1_R = std::sqrt(rho_R / p_R);
            DataType z2_L = z1_L * u_L;
            DataType z2_R = z1_R * u_R;
            DataType z3_L = z1_L * p_L;
            DataType z3_R = z1_R * p_R;
            DataType z2_bar = 0.5 * (z2_L + z2_R);
            DataType z3_bar_log;

            if (std::abs(z3_R - z3_L) / (std::abs(z3_L) + 1e-12) < 1e-6)
            {
                z3_bar_log = z3_L;
            }
            else
            {
                z3_bar_log = (z3_R - z3_L) / (std::log(z3_R) - std::log(z3_L));
            }

            DataType fs_1 = z2_bar * z3_bar_log;
            rhs_predict[isp][0] -=
                2.0 * Constant_s.getDMatrix()[isp][jsp] * fs_1 / local_det_jac;

            DataType z3_bar = 0.5 * (z3_L + z3_R);
            DataType z1_bar = 0.5 * (z1_L + z1_R);

            DataType fs_2 = z3_bar / z1_bar + z2_bar / z1_bar * fs_1;

            rhs_predict[isp][1] -=
                2.0 * Constant_s.getDMatrix()[isp][jsp] * fs_2 / local_det_jac;

            DataType z1_bar_log;
            if (std::abs(z1_R - z1_L) / (std::abs(z1_L) + 1e-12) < 1e-6)
            {
                z1_bar_log = z1_L;
            }
            else
            {
                z1_bar_log = (z1_R - z1_L) / (std::log(z1_R) - std::log(z1_L));
            }
            DataType fs_3 =
                0.5 * z2_bar / z1_bar *
                ((GAMMA + 1.0) / (GAMMA - 1.0) * z3_bar_log / z1_bar_log +
                 fs_2);
            rhs_predict[isp][2] -=
                2.0 * Constant_s.getDMatrix()[isp][jsp] * fs_3 / local_det_jac;
        }
    }
#endif
}

void Solver::compPredictionCR(const DataType (&consrv)[NSP][NCONSRV],
                              const DataType (&consrv_grad)[NSP][NCONSRV],
                              DataType (&rhs_predict)[NSP][NCONSRV])
{

    for (int isp = 0; isp < NSP; isp++)
    {
        DataType rho = consrv[isp][0];
        DataType rho_u = consrv[isp][1];
        DataType rho_E = consrv[isp][2];
        DataType d_rho = consrv_grad[isp][0];
        DataType d_rhou = consrv_grad[isp][1];
        DataType d_rhoE = consrv_grad[isp][2];
        DataType u = rho_u / rho;

        rhs_predict[isp][0] -= d_rhou;
        rhs_predict[isp][1] -= (-0.5 * u * u * (3.0 - GAMMA)) * d_rho +
                               u * (3.0 - GAMMA) * d_rhou +
                               (GAMMA - 1.0) * d_rhoE;
        rhs_predict[isp][2] -=
            (-rho_u * rho_E / rho / rho * GAMMA + (GAMMA - 1.0) * u * u * u) *
                d_rho +
            (rho_E / rho * GAMMA - 1.5 * (GAMMA - 1.0) * u * u) * d_rhou +
            GAMMA * u * d_rhoE;
    }
}

void Solver::computeElemRhsFR(Rhs *rhs_pool, const Element *elem_pool, int iele)
{
    const Element &element = elem_pool[iele];
    DataType local_det_jac = geom_pool[iele].local_det_jac;
#ifdef NS
    const Element &element_l = elem_pool[iele == 0 ? iele : iele - 1];
#else
    const Element &element_l =
        elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
#endif
    DataType local_det_jac_L =
        geom_pool[(iele - 1 + config.n_ele) % config.n_ele].local_det_jac;
#ifdef NS
    const Element &element_r =
        elem_pool[iele == config.n_ele - 1 ? iele : iele + 1];
#else
    const Element &element_r =
        elem_pool[(iele + 1 + config.n_ele) % config.n_ele];
#endif
    DataType local_det_jac_R =
        geom_pool[(iele + 1 + config.n_ele) % config.n_ele].local_det_jac;

    DataType flux_tmp[NSP][NCONSRV]{};
    DataType visflux_tmp[NSP][NCONSRV]{};

    for (int isp = 0; isp < NSP; isp++)
    {
        computeFlux(element.u_consrv[isp], flux_tmp[isp], config.a);
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            visflux_tmp[isp][ivar] = DataType(0.0);
        }
#ifdef LAD
        computeVisFlux(element.u_consrv[isp], element.u_grad_consrv[isp],
                       visflux_tmp[isp], config.nu);
#endif
#ifdef NS
        computeVisFlux(element.u_consrv[isp], element.u_grad_consrv[isp],
                       visflux_tmp[isp], config.vis);
#endif
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux_tmp[isp][ivar] -= visflux_tmp[isp][ivar];
        }
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

    if (config.enable_entropy_modify == true)
    {
        compPredictionEntropy(flux_tmp, element.u_consrv, local_det_jac,
                              rhs_prediction);
        // for (int isp = 0; isp < NSP; isp++)
        // {
        //     if (iele == 31)
        //         printf("iele=%d,flux_tmp[%d]=%f,rhs_predition[%d]=%f\n",
        //         iele,
        //                isp, flux_tmp[isp][0], isp, rhs_prediction[isp][0]);
        // }
    }
    else
    {
        compPredictionLP(flux_tmp, local_det_jac, rhs_prediction);
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

    DataType common_visflux_left[NCONSRV];
    DataType common_visflux_right[NCONSRV];
    DataType globalLift[NSP * NCONSRV] = {0.0};
    DataType globalLift_L[NSP * NCONSRV];
    DataType globalLift_R[NSP * NCONSRV];

    computeBR2Flux(element_l.u_consrv[ORDER], element_l.u_grad_consrv[ORDER],
                   element.u_consrv[0], element.u_grad_consrv[0],
                   local_det_jac_L, local_det_jac, common_visflux_left,
                   globalLift_L, globalLift, Constant_s, config.vis);
    computeBR2Flux(element.u_consrv[ORDER], element.u_grad_consrv[ORDER],
                   element_r.u_consrv[0], element_r.u_grad_consrv[0],
                   local_det_jac, local_det_jac_R, common_visflux_right,
                   globalLift, globalLift_R, Constant_s, config.vis);

    DataType flux_tmp2[NSP][NCONSRV]{};
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {

        flux_tmp2[0][ivar] =
            -1.0 * (common_flux_left[ivar] - common_visflux_left[ivar] -
                    flux_tmp[0][ivar]);
        flux_tmp2[ORDER][ivar] =
            1.0 * (common_flux_right[ivar] - common_visflux_right[ivar] -
                   flux_tmp[ORDER][ivar]);
    }
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                if (config.enable_entropy_modify == true)
                {
                    rhs_correction[isp][ivar] -=
                        invertMatrix<DataType, ORDER>(
                            getMMatrixEntropy<DataType, ORDER>())[isp][jsp] *
                        flux_tmp2[jsp][ivar] / local_det_jac;
                }
                else
                {
                    rhs_correction[isp][ivar] -=
                        invertMatrix<DataType, ORDER>(
                            Constant_s.getMMatrix())[isp][jsp] *
                        flux_tmp2[jsp][ivar] / local_det_jac;
                }
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
void Solver::computeRhs(Rhs *rhs_pool, const Element *elem_pool)
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

void Solver::compGradAndAvg()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        computeElementGrad(iele);
        ComputeElementAvg(iele);
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

    bool islimited[config.n_ele];
    DataType c1[config.n_ele][NCONSRV];
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Element &element = elem_pool_old[TORDER][iele];
        DataType hj = geom_pool[iele].x[1] - geom_pool[iele].x[0];
        Element &element_l =
            elem_pool_old[TORDER][(iele - 1 + config.n_ele) % config.n_ele];
        Element &element_r =
            elem_pool_old[TORDER][(iele + 1 + config.n_ele) % config.n_ele];

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
        Element &element = elem_pool_old[TORDER][iele];
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

    compGradAndAvg();
}

void Solver::timeNewExplicitSchemeK1()
{
    // compute u*_i
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[0][iele].u_consrv[isp][ivar] =
                    elem_pool_old[1][iele].u_consrv[isp][ivar];
                elem_pool_tmp[1][iele].u_consrv[isp][ivar] =
                    -elem_pool_old[0][iele].u_consrv[isp][ivar] +
                    2.0 * elem_pool_old[1][iele].u_consrv[isp][ivar];

                elem_pool_tmp[0][iele].u_grad_consrv[isp][ivar] =
                    elem_pool_old[1][iele].u_grad_consrv[isp][ivar];
                elem_pool_tmp[1][iele].u_grad_consrv[isp][ivar] =
                    -elem_pool_old[0][iele].u_grad_consrv[isp][ivar] +
                    2.0 * elem_pool_old[1][iele].u_grad_consrv[isp][ivar];
            }
        }
    }

    computeRhs(rhs_pool_tmp[0], elem_pool_tmp[0]);
    computeRhs(rhs_pool_tmp[1], elem_pool_tmp[1]);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[0][iele].u_consrv[isp][ivar] =
                    elem_pool_old[0][iele].u_consrv[isp][ivar];
                elem_pool_tmp[1][iele].u_consrv[isp][ivar] =
                    elem_pool_old[1][iele].u_consrv[isp][ivar];
            }
        }
    }

    DataType w = config.weight;
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                DataType u_nm1_0 = elem_pool_tmp[0][iele].u_consrv[isp][ivar];
                DataType u_nm1_1 = elem_pool_tmp[1][iele].u_consrv[isp][ivar];
                DataType rhs0 = rhs_pool_tmp[0][iele].rhs[isp][ivar];
                DataType rhs1 = rhs_pool_tmp[1][iele].rhs[isp][ivar];
                elem_pool_old[0][iele].u_consrv[isp][ivar] =
                    (1.0 / w - 1.0) * u_nm1_0 + (2.0 - 1.0 / w) * u_nm1_1 +

                    (rhs0 * (2.0 / w - 1.0) - rhs1) * config.dt * 0.5;
                elem_pool_old[1][iele].u_consrv[isp][ivar] =
                    1.0 * u_nm1_1 +

                    (rhs0 + rhs1) * config.dt * 0.5;
            }
        }
    }

    compGradAndAvg();

    std::ofstream ofile("monitor_point.csv", std::ios::app);
    if (!ofile)
    {
        std::cerr << "Error: Failed to create file." << std::endl;
        return;
    }
    ofile << elem_pool_old[0][0].u_consrv[0][0] << ","
          << elem_pool_old[1][0].u_consrv[0][0] << std::endl;
}

void Solver::timeNewExplicitSchemeKN()
{
    DataType alpha = DataType(1.0);
    Constant_t.valueTMatrix(alpha);
    auto Tmatrix = Constant_t.getTMatrix();
    auto Smatrix = Constant_t.getSMatrix();

    Eigen::MatrixXd Mat1(NTP, NTP);
    Eigen::MatrixXd Mat2(NTP, NTP);

    for (int itp = 0; itp < NTP; itp++)
    {
        for (int jtp = 0; jtp < NTP; jtp++)
        {
            Mat1(itp, jtp) = Smatrix[jtp][itp];
            Mat2(itp, jtp) = Tmatrix[itp][jtp];
        }
    }
    Mat2 = (1.0 - config.weight) * Mat1 * Mat2;
    Mat1 = (-config.weight) * Mat1;

    Mat1(TORDER, TORDER) += DataType(1.0);
    Mat2(0, TORDER) += DataType(1.0);

    Eigen::MatrixXd Matinv1 = Mat1.inverse();

    Mat2 = Matinv1 * Mat2;

    // compute u*_i
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int i = 0; i < NTP; i++)
                {
                    elem_pool_tmp[i][iele].u_consrv[isp][ivar] = 0.0;
                    elem_pool_tmp[i][iele].u_grad_consrv[isp][ivar] = 0.0;
                    for (int j = 0; j < NTP; j++)
                    {
                        elem_pool_tmp[i][iele].u_consrv[isp][ivar] +=
                            elem_pool_old[j][iele].u_consrv[isp][ivar] *
                            Tmatrix[i][j];

                        elem_pool_tmp[i][iele].u_grad_consrv[isp][ivar] +=
                            elem_pool_old[j][iele].u_grad_consrv[isp][ivar] *
                            Tmatrix[i][j];
                    }
                }
            }
        }
    }

    for (int itp = 0; itp < NTP; itp++)
    {
        computeRhs(rhs_pool_tmp[itp], elem_pool_tmp[itp]);
    }

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int itp = 0; itp < NTP; itp++)
                {
                    elem_pool_tmp[itp][iele].u_consrv[isp][ivar] =
                        elem_pool_old[itp][iele].u_consrv[isp][ivar];
                }
            }
        }
    }

    DataType w = config.weight;
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int itp = 0; itp < NTP; itp++)
                {
                    elem_pool_old[itp][iele].u_consrv[isp][ivar] =
                        DataType(0.0);
                    for (int jtp = 0; jtp < NTP; jtp++)
                    {
                        elem_pool_old[itp][iele].u_consrv[isp][ivar] +=
                            Matinv1(itp, jtp) *
                                rhs_pool_tmp[jtp][iele].rhs[isp][ivar] *
                                config.dt * 0.5 +
                            Mat2(itp, jtp) *
                                elem_pool_tmp[jtp][iele].u_consrv[isp][ivar];
                    }
                }
            }
        }
    }

    compGradAndAvg();

    std::ofstream ofile("monitor_point.csv", std::ios::app);
    if (!ofile)
    {
        std::cerr << "Error: Failed to create file." << std::endl;
        return;
    }
    ofile << elem_pool_old[0][0].u_consrv[0][0] << ","
          << elem_pool_old[1][0].u_consrv[0][0] << std::endl;
}

void Solver::timeRK1()
{

    computeRhs(rhs_pool_tmp[TORDER], elem_pool_old[TORDER]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +=
                    rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    compGradAndAvg();
}

void Solver::timeRK2()
{

    computeRhs(rhs_pool_tmp[TORDER], elem_pool_old[TORDER]);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {

                elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] =
                    elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +
                    rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }
    compGradAndAvg();

    computeRhs(rhs_pool_tmp[TORDER], elem_pool_tmp[TORDER]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[TORDER][iele].u_consrv[isp][ivar] =
                    DataType(0.5) *
                        elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(0.5) *
                        elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(0.5) * rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] *
                        config.dt;
            }
        }
    }

    compGradAndAvg();
}

void Solver::timeRK3()
{
    computeRhs(rhs_pool_tmp[TORDER], elem_pool_old[TORDER]);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] =
                    elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +
                    rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp[TORDER], elem_pool_tmp[TORDER]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] =
                    DataType(0.75) *
                        elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(0.25) *
                        elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(0.25) * rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] *
                        config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp[TORDER], elem_pool_tmp[TORDER]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[TORDER][iele].u_consrv[isp][ivar] =
                    DataType(1.0) / DataType(3.0) *
                        elem_pool_old[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        elem_pool_tmp[TORDER][iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        rhs_pool_tmp[TORDER][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }
    compGradAndAvg();
}

void Solver::init_RK3(int itp)
{
    if (itp < 1)
    {
        return;
    }

    computeRhs(rhs_pool_tmp[itp - 1], elem_pool_old[itp - 1]);

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[itp - 1][iele].u_consrv[isp][ivar] =
                    elem_pool_old[itp - 1][iele].u_consrv[isp][ivar] +
                    rhs_pool_tmp[itp - 1][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp[itp - 1], elem_pool_tmp[itp - 1]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_tmp[itp - 1][iele].u_consrv[isp][ivar] =
                    DataType(0.75) *
                        elem_pool_old[itp - 1][iele].u_consrv[isp][ivar] +
                    DataType(0.25) *
                        elem_pool_tmp[itp - 1][iele].u_consrv[isp][ivar] +
                    DataType(0.25) *
                        rhs_pool_tmp[itp - 1][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }

    computeRhs(rhs_pool_tmp[itp - 1], elem_pool_tmp[itp - 1]);
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                elem_pool_old[itp][iele].u_consrv[isp][ivar] =
                    DataType(1.0) / DataType(3.0) *
                        elem_pool_old[itp - 1][iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        elem_pool_tmp[itp - 1][iele].u_consrv[isp][ivar] +
                    DataType(2.0) / DataType(3.0) *
                        rhs_pool_tmp[itp - 1][iele].rhs[isp][ivar] * config.dt;
            }
        }
    }
    compGradAndAvg();
}

void Solver::computeElementGrad(int ielem)
{
    for (int itp = 0; itp < NTP; itp++)
    {

        Element &element = elem_pool_old[itp][ielem];

        DataType local_det_jac = geom_pool[ielem].local_det_jac;

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                element.u_grad_consrv[isp][ivar] = DataType(0);
                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    element.u_grad_consrv[isp][ivar] +=
                        Constant_s.getDMatrix()[isp][jsp] *
                        element.u_consrv[jsp][ivar] / local_det_jac;
                }
            }
        }
    }
}
void Solver::ComputeElementAvg(int ielem)
{
    for (int itp = 0; itp < NTP; itp++)
    {
        Element &element = elem_pool_old[itp][ielem];
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
        const Element &elem = elem_pool_old[TORDER][iele];
        for (int isp = 0; isp < NSP; isp++)
        {
            ofile << geom_pool[iele].x[isp] << ",";
        }

        DataType elem_primtv[NSP][NPRIMTV];

        for (int isp = 0; isp < NSP; isp++)
        {

            DataType primtv[NPRIMTV];
            Consrv2Primtv(elem.u_consrv[isp], primtv);
            for (int ivar = 0; ivar < NPRIMTV; ivar++)
            {
                elem_primtv[isp][ivar] = primtv[ivar];
            }
        }

        ofile << std::scientific << std::setprecision(12);
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

void Solver::OutputAvg(const std::string &filename)
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
        const Element &elem = elem_pool_old[TORDER][iele];

        ofile << geom_pool[iele].x[0] + 0.5 * geom_pool[iele].dx << ",";

        DataType elem_primtv_ave[NPRIMTV];

        Consrv2Primtv(elem.u_avg, elem_primtv_ave);

        ofile << std::scientific << std::setprecision(12);
        for (int ivar = 0; ivar < NPRIMTV; ivar++)
        {
            ofile << elem_primtv_ave[ivar] << ",";
        }

        ofile << std::endl;
    }
    return;
};