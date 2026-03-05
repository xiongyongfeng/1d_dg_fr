#pragma once
#include "config.h"
#include "constants.h"
#include "physics.h"
#include <cmath>

/**
 * @brief Burgers 方程
 * u_t + (0.5 * u^2)_x = 0
 */
class PhysicsBurgers : public PhysicsModel
{
  public:
    // 单点通量计算
    void computeFlux(const DataType u[NCONSRV], DataType flux[NCONSRV],
                     const Config &config) const override
    {
        (void)config;
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = 0.5 * u[ivar] * u[ivar];
        }
    }

    void computeFluxNormal(const DataType u[NCONSRV], DataType flux[NCONSRV],
                           const Config &config, DataType normal) const override
    {
        (void)config;
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = 0.5 * u[ivar] * u[ivar] * normal;
        }
    }

    void computeRiemannFlux(const DataType uL[NCONSRV],
                            const DataType uR[NCONSRV], DataType flux[NCONSRV],
                            const Config &config,
                            DataType normal) const override
    {
        (void)config;
        // Local Lax-Friedrichs
        DataType uL_val = uL[0];
        DataType uR_val = uR[0];
        DataType F_L = 0.5 * uL_val * uL_val * normal;
        DataType F_R = 0.5 * uR_val * uR_val * normal;
        DataType alpha =
            std::max(std::abs(uL_val * normal), std::abs(uR_val * normal));

        flux[0] = DataType(0.5) * (F_L + F_R - alpha * (uR_val - uL_val));
    }

    void prim2cons(const DataType prim[NPRIMTV],
                   DataType cons[NCONSRV]) const override
    {
        cons[0] = prim[0];
    }

    void cons2prim(const DataType cons[NCONSRV],
                   DataType prim[NPRIMTV]) const override
    {
        prim[0] = cons[0];
    }

    void setInitialCondition(DataType u[NSP][NCONSRV], const DataType x[NSP],
                             const Config &config) const override
    {
        (void)config;
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                u[isp][ivar] = 0.5 + std::sin(x[isp]);
            }
        }
    }

    std::string name() const override { return "Burgers"; }

    bool hasEntropyModify() const override { return true; }

    void compPredictionEntropy(const DataType flux[NSP][NCONSRV],
                               const DataType consrv[NSP][NCONSRV],
                               const DataType local_det_jac,
                               DataType rhs_predict[NSP][NCONSRV],
                               const Config &config) const override
    {
        (void)config;
        (void)flux;
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    rhs_predict[isp][ivar] -=
                        2.0 * getDMatrix<DataType, ORDER>()[isp][jsp] * 1.0 /
                        6.0 *
                        (consrv[isp][ivar] * consrv[isp][ivar] +
                         consrv[jsp][ivar] * consrv[isp][ivar] +
                         consrv[jsp][ivar] * consrv[jsp][ivar]) /
                        local_det_jac;
                }
            }
        }
    }

    // 计算守恒变量在计算域中的体积分
    void computeDomainIntegral(const Element *elem_pool, const Geom *geom_pool,
                               int n_ele,
                               DataType integral[NCONSRV]) const override
    {
        // 初始化为0
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            integral[ivar] = DataType(0.0);
        }

        // 对每个单元求积分并累加
        // 使用LGL求积公式: ∫u dx = sum_i w_i * u_i * (dx/2)
        for (int iele = 0; iele < n_ele; iele++)
        {
            const Element &elem = elem_pool[iele];
            DataType local_det_jac = geom_pool[iele].local_det_jac;

            for (int isp = 0; isp < NSP; isp++)
            {
                DataType weight = getLGLWeights<DataType, ORDER>()[isp];
                for (int ivar = 0; ivar < NCONSRV; ivar++)
                {
                    integral[ivar] +=
                        weight * elem.u_consrv[isp][ivar] * local_det_jac;
                }
            }
        }
    }
};
