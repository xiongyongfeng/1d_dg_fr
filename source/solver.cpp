#include "solver.h"
#include "constants.h"
#include "element.h"
#include "macro.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <valarray>

Solver::Solver(const Config &config, int nelem) : config(config)
{
    this->physics = createPhysicsModel();

    this->elem_pool_old = new Element *[NTP];
    this->elem_pool_tmp = new Element *[NTP];
    this->rhs_pool_tmp = new Rhs *[NTP];
    this->geom_pool = new Geom[nelem];

    Element *elem_old_block = new Element[nelem * NTP];
    Element *elem_tmp_block = new Element[nelem * NTP];
    Rhs *rhs_tmp_block = new Rhs[nelem * NTP];

    for (int itp = 0; itp < NTP; ++itp)
    {
        elem_pool_old[itp] = &elem_old_block[itp * nelem];
        elem_pool_tmp[itp] = &elem_tmp_block[itp * nelem];
        rhs_pool_tmp[itp] = &rhs_tmp_block[itp * nelem];
    }
}

Solver::~Solver()
{
    delete[] elem_pool_old[0];
    delete[] elem_pool_tmp[0];
    delete[] rhs_pool_tmp[0];

    delete[] elem_pool_old;
    delete[] elem_pool_tmp;
    delete[] rhs_pool_tmp;
    delete[] geom_pool;
}

void Solver::Initialization()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        Geom &geom = geom_pool[iele];
        for (int isp = 0; isp < NSP; isp++)
        {
            DataType dx = (config.x1 - config.x0) / config.n_ele;
            geom.x[isp] = config.x0 + iele * dx + isp * dx / (NSP - 1);
        }
        geom.dx = geom.x[ORDER] - geom.x[0];
        geom.local_det_jac = geom.dx / 2.0;
    }

    for (int itp = 0; itp < NTP; itp++)
    {
        for (int iele = 0; iele < config.n_ele; iele++)
        {
            Element &elem = elem_pool_old[itp][iele];

            // 使用物理模型的初始条件
            physics->setInitialCondition(elem.u_consrv, geom_pool[iele].x,
                                         config);

            computeElementGrad(iele);
            ComputeElementAvg(iele);
        }
    }
}

void Solver::computeElemRhsDG(Rhs *rhs_pool, const Element *elem_pool, int iele)
{
    const Element &element = elem_pool[iele];
    DataType length_scale = geom_pool[iele].dx; // 计算特征长度尺度
    DataType local_det_jac = geom_pool[iele].local_det_jac;
    const Element &element_l =
        elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
    DataType local_det_jac_L =
        geom_pool[(iele - 1 + config.n_ele) % config.n_ele].local_det_jac;
    const Element &element_r =
        elem_pool[(iele + 1 + config.n_ele) % config.n_ele];
    DataType local_det_jac_R =
        geom_pool[(iele + 1 + config.n_ele) % config.n_ele].local_det_jac;

    DataType rhs_tmp[NSP][NCONSRV]{};
    DataType flux_tmp[NSP][NCONSRV]{};

    // 计算单元内部通量
    for (int isp = 0; isp < NSP; isp++)
    {
        physics->computeFlux(element.u_consrv[isp], flux_tmp[isp], config);
    }

    // 体通量项
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

    // 界面通量 - 左边界
    DataType rhs_common_flux_left[NCONSRV];
    DataType uL_bc[NCONSRV];
    DataType uL_grad_bc[NCONSRV];

    if (iele == 0 && config.bc_type != 0)
    {
        // Dirichlet边界: 使用边界值
        getBoundaryState(-1, uL_bc, uL_grad_bc);
        physics->computeRiemannFlux(uL_bc, element.u_consrv[0],
                                    rhs_common_flux_left, config);
    }
    else
    {
        physics->computeRiemannFlux(element_l.u_consrv[ORDER],
                                    element.u_consrv[0], rhs_common_flux_left,
                                    config);
    }

    // 界面通量 - 右边界
    DataType rhs_common_flux_right[NCONSRV];
    DataType uR_bc[NCONSRV];
    DataType uR_grad_bc[NCONSRV];

    if (iele == config.n_ele - 1 && config.bc_type != 0)
    {
        // Dirichlet边界: 使用边界值
        getBoundaryState(1, uR_bc, uR_grad_bc);
        physics->computeRiemannFlux(element.u_consrv[ORDER], uR_bc,
                                    rhs_common_flux_right, config);
    }
    else
    {
        physics->computeRiemannFlux(element.u_consrv[ORDER],
                                    element_r.u_consrv[0],
                                    rhs_common_flux_right, config);
    }

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        rhs_tmp[0][ivar] += rhs_common_flux_left[ivar];
        rhs_tmp[ORDER][ivar] -= rhs_common_flux_right[ivar];
    }

    // 扩散项 (如果有)
    if (physics->hasDiffusion() && config.vis_scheme_type == 0)
    {
        DataType visflux_tmp[NSP][NCONSRV]{};
        DataType rhs_common_visflux_left[NCONSRV];
        DataType rhs_common_visflux_right[NCONSRV];
        DataType globalLift[NSP * NCONSRV] = {0.0};
        DataType globalLift_L[NSP * NCONSRV];
        DataType globalLift_R[NSP * NCONSRV];

        physics->computeBR2Flux(
            element_l.u_consrv[ORDER], element_l.u_grad_consrv[ORDER],
            element.u_consrv[0], element.u_grad_consrv[0], local_det_jac_L,
            local_det_jac, rhs_common_visflux_left, globalLift_L, globalLift,
            config);
        physics->computeBR2Flux(
            element.u_consrv[ORDER], element.u_grad_consrv[ORDER],
            element_r.u_consrv[0], element_r.u_grad_consrv[0], local_det_jac,
            local_det_jac_R, rhs_common_visflux_right, globalLift, globalLift_R,
            config);

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_tmp[0][ivar] -= rhs_common_visflux_left[ivar];
            rhs_tmp[ORDER][ivar] += rhs_common_visflux_right[ivar];
        }

        // 计算修正后的梯度
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
            physics->computeVisFlux(element.u_consrv[isp], grad_u_consrv_[isp],
                                    visflux_tmp[isp], config);
        }

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    rhs_tmp[isp][ivar] -=
                        getSMatrix<DataType, ORDER>()[jsp][isp] *
                        visflux_tmp[jsp][ivar];
                }
            }
        }
    }
    else if (physics->hasDiffusion() && config.vis_scheme_type == 1)
    {
        // 处理其他粘性方案
        DataType visflux_tmp[NSP][NCONSRV]{};
        for (int isp = 0; isp < NSP; isp++)
        {
            physics->computeVisFlux(element.u_consrv[isp],
                                    element.u_grad_consrv[isp],
                                    visflux_tmp[isp], config);
        }

        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    rhs_tmp[isp][ivar] -=
                        getSMatrix<DataType, ORDER>()[jsp][isp] *
                        visflux_tmp[jsp][ivar];
                }
            }
        }

        DataType left_bc_u_out[NCONSRV];
        DataType left_bc_u_in[NCONSRV];
        DataType right_bc_u_out[NCONSRV];
        DataType right_bc_u_in[NCONSRV];
        for (size_t i = 0; i < NCONSRV; i++)
        {
            right_bc_u_out[i] = element_r.u_consrv[0][i];
            right_bc_u_in[i] = element.u_consrv[ORDER][i];
            left_bc_u_out[i] = element_l.u_consrv[ORDER][i];
            left_bc_u_in[i] = element.u_consrv[0][i];
        }

        // 粘性界面通量 - 左边界
        DataType rhs_vis_common_flux_left[NCONSRV];

        if (iele == 0 && config.bc_type != 0)
        {
            // Dirichlet边界
            physics->computeVisRiemannFlux(
                uL_bc, uL_grad_bc, element.u_consrv[0],
                element.u_grad_consrv[0], length_scale,
                rhs_vis_common_flux_left, config);
            for (size_t i = 0; i < NCONSRV; i++)
            {
                left_bc_u_out[i] = uL_bc[i];
            }
        }
        else
        {
            physics->computeVisRiemannFlux(
                element_l.u_consrv[ORDER], element_l.u_grad_consrv[ORDER],
                element.u_consrv[0], element.u_grad_consrv[0], length_scale,
                rhs_vis_common_flux_left, config);
        }

        // 粘性界面通量 - 右边界
        DataType rhs_vis_common_flux_right[NCONSRV];

        if (iele == config.n_ele - 1 && config.bc_type != 0)
        {
            // Dirichlet边界
            physics->computeVisRiemannFlux(
                element.u_consrv[ORDER], element.u_grad_consrv[ORDER], uR_bc,
                uR_grad_bc, length_scale, rhs_vis_common_flux_right, config);
            for (size_t i = 0; i < NCONSRV; i++)
            {
                right_bc_u_out[i] = uR_bc[i];
            }
        }
        else
        {
            physics->computeVisRiemannFlux(
                element.u_consrv[ORDER], element.u_grad_consrv[ORDER],
                element_r.u_consrv[0], element_r.u_grad_consrv[0], length_scale,
                rhs_vis_common_flux_right, config);
        }

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_tmp[0][ivar] += rhs_vis_common_flux_left[ivar];
            rhs_tmp[ORDER][ivar] -= rhs_vis_common_flux_right[ivar];
        }

        DataType rhs_nipg_left[NSP][NCONSRV];
        DataType rhs_nipg_right[NSP][NCONSRV];
        DataType beta = DataType(1.0);
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                rhs_nipg_right[isp][ivar] =
                    -DataType(0.5) * beta * config.nu *
                    (right_bc_u_out[ivar] - right_bc_u_in[ivar]) *
                    getDMatrix<DataType, ORDER>()[ORDER][isp] / local_det_jac;
                rhs_nipg_left[isp][ivar] =
                    -DataType(0.5) * beta * config.nu *
                    (left_bc_u_out[ivar] - left_bc_u_in[ivar]) *
                    getDMatrix<DataType, ORDER>()[0][isp] / local_det_jac *
                    DataType(-1.0);
            }
        }
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                rhs_tmp[isp][ivar] +=
                    rhs_nipg_left[isp][ivar] + rhs_nipg_right[isp][ivar];
            }
        }
    }
    else
    {
        // 无粘性项，无需处理
    }

    // 质量矩阵求逆
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_pool[iele].rhs[isp][ivar] = DataType(0.0);
            for (int jsp = 0; jsp < NSP; jsp++)
            {
                rhs_pool[iele].rhs[isp][ivar] +=
                    invertMatrix<DataType, ORDER>(
                        getMMatrix<DataType, ORDER>())[isp][jsp] *
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
                rhs_predict[isp][ivar] -=
                    getDMatrix<DataType, ORDER>()[isp][jsp] * flux[jsp][ivar] /
                    local_det_jac;
            }
        }
    }
}

void Solver::computeElemRhsFR(Rhs *rhs_pool, const Element *elem_pool, int iele)
{
    const Element &element = elem_pool[iele];
    DataType local_det_jac = geom_pool[iele].local_det_jac;
    const Element &element_l =
        elem_pool[(iele - 1 + config.n_ele) % config.n_ele];
    const Element &element_r =
        elem_pool[(iele + 1 + config.n_ele) % config.n_ele];

    DataType flux_tmp[NSP][NCONSRV]{};

    for (int isp = 0; isp < NSP; isp++)
    {
        physics->computeFlux(element.u_consrv[isp], flux_tmp[isp], config);
    }

    DataType rhs_prediction[NSP][NCONSRV]{};

    // 熵修正或标准LP预测
    if (config.enable_entropy_modify && physics->hasEntropyModify())
    {
        physics->compPredictionEntropy(flux_tmp, element.u_consrv,
                                       local_det_jac, rhs_prediction, config);
    }
    else
    {
        compPredictionLP(flux_tmp, local_det_jac, rhs_prediction);
    }

    // 界面通量
    DataType common_flux_left[NCONSRV];
    if (iele == 0 && config.bc_type != 0)
    {
        // Dirichlet边界
        DataType uL_bc[NCONSRV];
        DataType uL_grad_bc[NCONSRV];
        getBoundaryState(-1, uL_bc, uL_grad_bc);
        physics->computeRiemannFlux(uL_bc, element.u_consrv[0],
                                    common_flux_left, config);
    }
    else
    {
        physics->computeRiemannFlux(element_l.u_consrv[ORDER],
                                    element.u_consrv[0], common_flux_left,
                                    config);
    }

    DataType common_flux_right[NCONSRV];
    if (iele == config.n_ele - 1 && config.bc_type != 0)
    {
        // Dirichlet边界
        DataType uR_bc[NCONSRV];
        DataType uR_grad_bc[NCONSRV];
        getBoundaryState(1, uR_bc, uR_grad_bc);
        physics->computeRiemannFlux(element.u_consrv[ORDER], uR_bc,
                                    common_flux_right, config);
    }
    else
    {
        physics->computeRiemannFlux(element.u_consrv[ORDER],
                                    element_r.u_consrv[0], common_flux_right,
                                    config);
    }

    // 修正项
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
                auto matrix =
                    config.enable_entropy_modify && physics->hasEntropyModify()
                        ? getMMatrixEntropy<DataType, ORDER>()
                        : getMMatrix<DataType, ORDER>();
                rhs_correction[isp][ivar] -=
                    invertMatrix<DataType, ORDER>(matrix)[isp][jsp] *
                    flux_tmp2[jsp][ivar] / local_det_jac;
            }
        }
    }

    // 组装最终RHS
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            rhs_pool[iele].rhs[isp][ivar] =
                rhs_prediction[isp][ivar] + rhs_correction[isp][ivar];
        }
    }
}

void Solver::computeRhs(Rhs *rhs_pool, const Element *elem_pool)
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        if (config.dg_fr_type == 0)
        {
            computeElemRhsDG(rhs_pool, elem_pool, iele);
        }
        else if (config.dg_fr_type == 1)
        {
            computeElemRhsFR(rhs_pool, elem_pool, iele);
        }
    }
}

void Solver::compGradAndAvg()
{
    for (int iele = 0; iele < config.n_ele; iele++)
    {
        computeElementGrad(iele);
        ComputeElementAvg(iele);
    }
}

std::pair<DataType, int> Solver::Minmod(DataType a, DataType b, DataType c)
{
    if (a > -1e-6 && a < 1e-6)
        return {DataType(a), 0};

    const bool all_nonneg = (a >= 0) && (b >= 0) && (c >= 0);
    const bool all_nonpos = (a <= 0) && (b <= 0) && (c <= 0);

    if (!(all_nonneg || all_nonpos))
    {
        return {DataType(0), 1};
    }

    const DataType min_abs = std::min({std::abs(a), std::abs(b), std::abs(c)});
    const DataType result = (a > 0) ? min_abs : -min_abs;

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
    // 计算 u*_i
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
                    1.0 * u_nm1_1 + (rhs0 + rhs1) * config.dt * 0.5;
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
                        getDMatrix<DataType, ORDER>()[isp][jsp] *
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
            physics->cons2prim(elem.u_consrv[isp], primtv);
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
}

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
        physics->cons2prim(elem.u_avg, elem_primtv_ave);

        for (int ivar = 0; ivar < NPRIMTV; ivar++)
        {
            ofile << elem_primtv_ave[ivar] << ",";
        }
        ofile << std::endl;
    }
}

void Solver::computeCflDt()
{
    if (config.cfl <= 0.0)
    {
        return; // 如果cfl <= 0，不自动计算dt
    }

    DataType min_dt = 1e20;

    for (int iele = 0; iele < config.n_ele; iele++)
    {
        DataType dx = geom_pool[iele].dx;
        DataType local_det_jac = geom_pool[iele].local_det_jac;
        const Element &elem = elem_pool_old[TORDER][iele];

        // 获取原始变量用于计算特征值
        DataType prim[NPRIMTV];
        physics->cons2prim(elem.u_avg, prim);

        DataType dt_elem = 1e20;

        // 平流项CFL条件: dt <= cfl * dx / |lambda_max|
        // 对于不同方程，lambda_max不同
        DataType lambda_max = 0.0;

        if (physics->name() == "LAD")
        {
            // 线性平流: lambda = a
            lambda_max = std::abs(config.a);
        }
        else if (physics->name() == "Burgers")
        {
            // Burgers: lambda = u
            lambda_max = std::abs(prim[0]);
        }
        else if (physics->name() == "NS")
        {
            // Euler/NS: lambda = |u| + c
            DataType c = std::sqrt(GAMMA * prim[2] / prim[0]);
            lambda_max = std::abs(prim[1]) + c;
        }

        if (lambda_max > 1e-10)
        {
            DataType dt_conv =
                config.cfl * dx / (2.0 * ORDER + 1.0) / lambda_max;
            dt_elem = std::min(dt_elem, dt_conv);
        }

        // 扩散项CFL条件: dt <= cfl * dx^2 / (4 * nu)
        // 对于二阶扩散项，显式格式的稳定性条件
        if (physics->hasDiffusion() && config.nu > 1e-10)
        {
            DataType dt_diff = config.cfl * dx * dx / (2.0 * ORDER + 1.0) /
                               (2.0 * ORDER + 1.0) / config.nu;
            dt_elem = std::min(dt_elem, dt_diff);
        }

        min_dt = std::min(min_dt, dt_elem);
    }

    config.dt = min_dt;
    std::cout << "CFL: " << config.cfl << ", Computed dt: " << config.dt
              << std::endl;
}

void Solver::getBoundaryState(int bc_pos, DataType u_bc[NCONSRV],
                              DataType u_grad_bc[NCONSRV]) const
{
    // bc_pos: -1 = 左边界, +1 = 右边界

    if (config.bc_type == 0)
    {
        // Periodic边界: 使用对端单元的值
        if (bc_pos == -1)
        {
            // 左边界: 使用最后一个单元的值
            const Element &elem_right = elem_pool_old[TORDER][config.n_ele - 1];
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                u_bc[ivar] = elem_right.u_consrv[ORDER][ivar];
                u_grad_bc[ivar] = elem_right.u_grad_consrv[ORDER][ivar];
            }
        }
        else
        {
            // 右边界: 使用第一个单元的值
            const Element &elem_left = elem_pool_old[TORDER][0];
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                u_bc[ivar] = elem_left.u_consrv[0][ivar];
                u_grad_bc[ivar] = elem_left.u_grad_consrv[0][ivar];
            }
        }
    }
    else
    {
        // Dirichlet边界: 使用给定的边界值
        DataType bc_value = (bc_pos == -1) ? config.bc_left : config.bc_right;

        // 将边界值作为原始变量，然后转换为守恒变量
        DataType prim[NPRIMTV] = {bc_value};
        DataType u_b_input[NCONSRV];
        physics->prim2cons(prim, u_b_input);

        if (bc_pos == -1)
        {
            // 左边界: 使用最后一个单元的值
            const Element &elem_right = elem_pool_old[TORDER][0];
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                u_bc[ivar] = u_b_input[ivar] * DataType(2.0) -
                             elem_right.u_consrv[0][ivar];
                u_grad_bc[ivar] = elem_right.u_grad_consrv[0][ivar];
            }
        }
        else
        {
            // 右边界: 使用第一个单元的值
            const Element &elem_left = elem_pool_old[TORDER][config.n_ele - 1];
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                u_bc[ivar] = u_b_input[ivar] * DataType(2.0) -
                             elem_left.u_consrv[ORDER][ivar];
                u_grad_bc[ivar] = elem_left.u_grad_consrv[ORDER][ivar];
            }
        }
    }
}
