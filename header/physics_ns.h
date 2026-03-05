#pragma once
#include "config.h"
#include "constants.h"
#include "physics.h"
#include <cmath>

/**
 * @brief 一维Euler方程 (Navier-Stokes 无粘项)
 * [rho, rho*u, rho*E]_t + [rho*u, rho*u^2+p, (rho*E+p)*u]_x = 0
 */
class PhysicsNS : public PhysicsModel
{
  public:
    // 单点通量计算
    void computeFlux(const DataType u[NCONSRV], DataType flux[NCONSRV],
                     const Config &config) const override
    {
        (void)config;
        DataType prim[NPRIMTV];
        cons2prim(u, prim);
        flux[0] = prim[0] * prim[1];
        flux[1] = prim[0] * prim[1] * prim[1] + prim[2];
        flux[2] = (u[2] + prim[2]) * prim[1];
    }

    void computeFluxNormal(const DataType u[NCONSRV], DataType flux[NCONSRV],
                           const Config &config, DataType normal) const override
    {
        (void)config;
        DataType prim[NPRIMTV];
        cons2prim(u, prim);
        DataType un = prim[1] * normal;
        flux[0] = prim[0] * un;
        flux[1] = prim[0] * prim[1] * un + prim[2] * normal;
        flux[2] = (u[2] + prim[2]) * un;
    }

    void computeRiemannFlux(const DataType uL[NCONSRV],
                            const DataType uR[NCONSRV], DataType flux[NCONSRV],
                            const Config &config,
                            DataType normal) const override
    {
        (void)config;
        DataType primtv_l[NPRIMTV];
        DataType primtv_r[NPRIMTV];
        cons2prim(uL, primtv_l);
        cons2prim(uR, primtv_r);
        DataType c_L = GetSoundSpeed(primtv_l);
        DataType c_R = GetSoundSpeed(primtv_r);

        DataType un_l = primtv_l[1] * normal;
        DataType un_r = primtv_r[1] * normal;
        DataType flux_l[NCONSRV];
        DataType flux_r[NCONSRV];
        computeFluxNormal(uL, flux_l, config, normal);
        computeFluxNormal(uR, flux_r, config, normal);

        // HLL
        DataType lambda_L = std::min(un_l - c_L, un_r - c_R);
        DataType lambda_R = std::max(un_l + c_L, un_r + c_R);

        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {
            if (lambda_L >= 0)
            {
                flux[ivar] = flux_l[ivar];
            }
            else if (lambda_R <= 0)
            {
                flux[ivar] = flux_r[ivar];
            }
            else
            {
                flux[ivar] =
                    (lambda_R * flux_l[ivar] - lambda_L * flux_r[ivar] +
                     lambda_L * lambda_R * (uR[ivar] - uL[ivar])) /
                    (lambda_R - lambda_L);
            }
        }
    }

    void prim2cons(const DataType prim[NPRIMTV],
                   DataType cons[NCONSRV]) const override
    {
        cons[0] = prim[0];
        cons[1] = prim[0] * prim[1];
        cons[2] = 0.5 * prim[0] * prim[1] * prim[1] + prim[2] / (GAMMA - 1);
    }

    void cons2prim(const DataType cons[NCONSRV],
                   DataType prim[NPRIMTV]) const override
    {
        prim[0] = cons[0];
        prim[1] = cons[1] / cons[0];
        prim[2] = (cons[2] - 0.5 * prim[0] * prim[1] * prim[1]) * (GAMMA - 1);
        prim[3] = prim[2] / (prim[0] * GAS_R);
    }

    void setInitialCondition(DataType u[NSP][NCONSRV], const DataType x[NSP],
                             const Config &config) const override
    {
        (void)config;
        DataType consrv_left[NCONSRV];
        DataType consrv_right[NCONSRV];
        DataType primtv_left[NPRIMTV];
        DataType primtv_right[NPRIMTV];

        // Sod problem initial condition
        primtv_left[0] = 1.0;
        primtv_left[1] = 0.0;
        primtv_left[2] = 1.0;
        primtv_left[3] = 1.0;

        primtv_right[0] = 0.125;
        primtv_right[1] = 0.0;
        primtv_right[2] = 0.1;
        primtv_right[3] = 0.1 / 0.125;

        prim2cons(primtv_left, consrv_left);
        prim2cons(primtv_right, consrv_right);

        for (int isp = 0; isp < NSP; isp++)
        {
            if (x[isp] < DataType(0.0))
            {
                u[isp][0] = consrv_left[0];
                u[isp][1] = consrv_left[1];
                u[isp][2] = consrv_left[2];
            }
            else
            {
                u[isp][0] = consrv_right[0];
                u[isp][1] = consrv_right[1];
                u[isp][2] = consrv_right[2];
            }
        }
    }

    std::string name() const override { return "NS"; }

    bool hasEntropyModify() const override { return true; }

    void compPredictionEntropy(const DataType flux[NSP][NCONSRV],
                               const DataType consrv[NSP][NCONSRV],
                               const DataType local_det_jac,
                               DataType rhs_predict[NSP][NCONSRV],
                               const Config &config) const override
    {
        (void)config;
        (void)flux;

        // Ismail & Roe entropy stable flux
        DataType primtv[NSP][NPRIMTV];
        for (int isp = 0; isp < NSP; isp++)
        {
            cons2prim(consrv[isp], primtv[isp]);
        }

        for (int isp = 0; isp < NSP; isp++)
        {
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
                DataType z3_bar_log = z3_L;
                if (std::abs(z3_R - z3_L) / (std::abs(z3_L) + 1e-12) >= 1e-6)
                {
                    z3_bar_log =
                        (z3_R - z3_L) / (std::log(z3_R) - std::log(z3_L));
                }

                DataType fs_1 = z2_bar * z3_bar_log;
                rhs_predict[isp][0] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_1 / local_det_jac;

                DataType z3_bar = 0.5 * (z3_L + z3_R);
                DataType z1_bar = 0.5 * (z1_L + z1_R);

                DataType fs_2 = z3_bar / z1_bar + z2_bar / z1_bar * fs_1;
                rhs_predict[isp][1] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_2 / local_det_jac;

                DataType z1_bar_log = z1_L;
                if (std::abs(z1_R - z1_L) / (std::abs(z1_L) + 1e-12) >= 1e-6)
                {
                    z1_bar_log =
                        (z1_R - z1_L) / (std::log(z1_R) - std::log(z1_L));
                }
                DataType fs_3 =
                    0.5 * z2_bar / z1_bar *
                    ((GAMMA + 1.0) / (GAMMA - 1.0) * z3_bar_log / z1_bar_log +
                     fs_2);
                rhs_predict[isp][2] -= 2.0 *
                                       getDMatrix<DataType, ORDER>()[isp][jsp] *
                                       fs_3 / local_det_jac;
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

  private:
    // Helper functions
    DataType GetSoundSpeed(const DataType prim[NPRIMTV]) const
    {
        return GAMMA * prim[2] / prim[0];
    }
};
