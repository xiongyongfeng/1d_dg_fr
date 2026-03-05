#pragma once
#include "config.h"
#include "constants.h"
#include "physics.h"
#include <cmath>

/**
 * @brief 线性平流扩散方程 (Linear Advection Diffusion)
 * u_t + a * u_x = nu * u_xx
 */
class PhysicsLAD : public PhysicsModel
{
  public:
    // 单点通量计算
    void computeFlux(const DataType u[NCONSRV], DataType flux[NCONSRV],
                     const Config &config) const override
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = config.a * u[ivar];
        }
    }

    void computeFluxNormal(const DataType u[NCONSRV], DataType flux[NCONSRV],
                           const Config &config, DataType normal) const override
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = config.a * u[ivar] * normal;
        }
    }

    void computeRiemannFlux(const DataType uL[NCONSRV],
                            const DataType uR[NCONSRV], DataType flux[NCONSRV],
                            const Config &config,
                            DataType normal) const override
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            DataType F_L = config.a * uL[ivar] * normal;
            DataType F_R = config.a * uR[ivar] * normal;
            DataType alpha = std::abs(config.a * normal);
            // Lax-Friedrichs 通量
            flux[ivar] =
                DataType(0.5) * (F_L + F_R - alpha * (uR[ivar] - uL[ivar]));
        }
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
                // u[isp][ivar] = std::sin(2 * acos(-1.0) * x[isp]);

                u[isp][ivar] = config.bc_right;

                // if (x[isp] < DataType(0.25))
                // {
                //     u[isp][ivar] = DataType(10.0);
                // }
                // else if (x[isp] > DataType(0.75))
                // {
                //     u[isp][ivar] = DataType(10.0);
                // }
                // else
                // {
                //     u[isp][ivar] = DataType(11.0);
                // }
            }
        }
    }

    std::string name() const override { return "LAD"; }

    bool hasDiffusion() const override { return true; }

    void computeVisFlux(const DataType u[NCONSRV],
                        const DataType u_grad[NCONSRV], DataType flux[NCONSRV],
                        const Config &config) const override
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] = config.nu * u_grad[ivar];
        }
    }

    // 粘性通量黎曼求解器 - 使用中心通量 (LAD方程)
    void computeVisRiemannFlux(const DataType uL[NCONSRV],
                               const DataType uL_grad[NCONSRV],
                               const DataType uR[NCONSRV],
                               const DataType uR_grad[NCONSRV],
                               const DataType &length_scale,
                               DataType flux[NCONSRV], const Config &config,
                               DataType normal) const override
    {
        // 中心通量: F* = 0.5 * (F_L + F_R)
        // 其中 F_L = nu * uL_grad, F_R = nu * uR_grad
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            DataType flux_L = config.nu * uL_grad[ivar] * normal;
            DataType flux_R = config.nu * uR_grad[ivar] * normal;
            flux[ivar] = DataType(0.5) * (flux_L + flux_R) -
                         config.ip_coef * (2.0 * ORDER + 1.0) * config.nu *
                             (uL[ivar] - uR[ivar]) / length_scale;
        }
    }

    void
    computeBR2Flux(const DataType uL[NCONSRV], const DataType uL_grad[NCONSRV],
                   const DataType uR[NCONSRV], const DataType uR_grad[NCONSRV],
                   DataType local_det_jac_L, DataType local_det_jac_R,
                   DataType flux[NCONSRV], DataType globalLift_L[NSP * NCONSRV],
                   DataType globalLift_R[NSP * NCONSRV],
                   const Config &config) const override
    {
        auto Mass_inv =
            invertMatrix<DataType, ORDER>(getMMatrix<DataType, ORDER>());
        auto MEntropy = getMMatrixEntropy<DataType, ORDER>();

        // step1 : compute local lift operator
        DataType localLift_L[NSP][NCONSRV];
        DataType localLift_R[NSP][NCONSRV];
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                DataType jump = uR[ivar] - uL[ivar];

                localLift_L[isp][ivar] = DataType(0.0);
                localLift_R[isp][ivar] = DataType(0.0);

                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    localLift_L[isp][ivar] += Mass_inv[isp][jsp] *
                                              MEntropy[isp][ORDER] * jump *
                                              (-0.5) / local_det_jac_L;

                    localLift_R[isp][ivar] += Mass_inv[isp][jsp] *
                                              MEntropy[isp][0] * jump * (-0.5) /
                                              local_det_jac_R;
                }
            }
        }

        // step2 : add contribution to global lift operator
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                globalLift_L[isp * NCONSRV + ivar] += localLift_L[isp][ivar];
                globalLift_R[isp * NCONSRV + ivar] += localLift_R[isp][ivar];
            }
        }

        // step3 : compute invflux
        DataType uL_grad_[NCONSRV];
        DataType uR_grad_[NCONSRV];

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            uL_grad_[ivar] = localLift_L[ORDER][ivar];
            uR_grad_[ivar] = localLift_R[0][ivar];
        }

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            flux[ivar] =
                DataType(0.5) * config.nu * (uL_grad[ivar] - uL_grad_[ivar]) +
                DataType(0.5) * config.nu * (uR_grad[ivar] - uR_grad_[ivar]);
        }
    }

    bool hasEntropyModify() const override { return true; }

    void compPredictionEntropy(const DataType flux[NSP][NCONSRV],
                               const DataType consrv[NSP][NCONSRV],
                               const DataType local_det_jac,
                               DataType rhs_predict[NSP][NCONSRV],
                               const Config &config) const override
    {
        (void)config;
        (void)consrv;
        for (int isp = 0; isp < NSP; isp++)
        {
            for (int ivar = 0; ivar < NCONSRV; ivar++)
            {
                for (int jsp = 0; jsp < NSP; jsp++)
                {
                    rhs_predict[isp][ivar] -=
                        2.0 * getDMatrix<DataType, ORDER>()[isp][jsp] * 0.5 *
                        (flux[jsp][ivar] + flux[isp][ivar]) / local_det_jac;
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
