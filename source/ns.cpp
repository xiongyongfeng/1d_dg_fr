#include "ns.h"
#include "config.h"
#include "constants.h"
#include "macro.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <math.h>

void Primtv2Consrv(const DataType (&primtv)[NPRIMTV],
                   DataType (&consrv)[NCONSRV])
{
    consrv[0] = primtv[0];
    consrv[1] = primtv[0] * primtv[1];
    consrv[2] =
        0.5 * primtv[0] * primtv[1] * primtv[1] + primtv[2] / (GAMMA - 1);
}

void Consrv2Primtv(const DataType (&consrv)[NCONSRV],
                   DataType (&primtv)[NPRIMTV])
{
    primtv[0] = consrv[0];
    primtv[1] = consrv[1] / consrv[0];
    primtv[2] =
        (consrv[2] - 0.5 * primtv[0] * primtv[1] * primtv[1]) * (GAMMA - 1);
    primtv[3] = primtv[2] / (primtv[0] * GAS_R);
}

DataType GetSoundSpeed(const DataType (&consrv)[NCONSRV],
                       const DataType (&primtv)[NPRIMTV])
{
    return GAMMA * primtv[2] / primtv[0];
}

void computeRiemannFlux(const DataType (&uL)[NCONSRV],
                        const DataType (&uR)[NCONSRV],
                        DataType (&flux)[NCONSRV], DataType a)
{
    DataType primtv_l[NPRIMTV];
    DataType primtv_r[NPRIMTV];
    Consrv2Primtv(uL, primtv_l);
    Consrv2Primtv(uR, primtv_r);
    DataType flux_l[NCONSRV];
    DataType flux_r[NCONSRV];
    computeFlux(uL, flux_l);
    computeFlux(uR, flux_r);

    // Local LF
    //  DataType lambda_l = std::fabs(primtv_l[1]) + GetSoundSpeed(uL,
    //  primtv_l); DataType lambda_r = std::fabs(primtv_r[1]) +
    //  GetSoundSpeed(uR, primtv_r); DataType lambda = std::max(lambda_l,
    //  lambda_r); for (int ivar = 0; ivar < NCONSRV; ivar++)
    //  {
    //      flux[ivar] = 0.5 * (flux_l[ivar] + flux_r[ivar]) -
    //                   0.5 * lambda * (uR[ivar] - uL[ivar]);
    //  }

    // HLL
    DataType c_L = GetSoundSpeed(uL, primtv_l);
    DataType c_R = GetSoundSpeed(uR, primtv_r);
    DataType lambda_L = std::min(primtv_l[1] - c_L, primtv_r[1] - c_R);
    DataType lambda_R = std::max(primtv_l[1] + c_L, primtv_r[1] + c_R);
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
            flux[ivar] = (lambda_R * flux_l[ivar] - lambda_L * flux_r[ivar] +
                          lambda_L * lambda_R * (uR[ivar] - uL[ivar])) /
                         (lambda_R - lambda_L);
        }
    }
}

void computeFlux(const DataType (&u)[NCONSRV], DataType (&flux)[NCONSRV],
                 DataType a)
{
    DataType primtv[NPRIMTV];
    Consrv2Primtv(u, primtv);
    flux[0] = primtv[0] * primtv[1];
    flux[1] = primtv[0] * primtv[1] * primtv[1] + primtv[2];
    flux[2] = (u[2] + primtv[2]) * primtv[1];
}

void computeVisFlux(const DataType (&u)[NCONSRV],
                    const DataType (&u_grad)[NCONSRV],
                    DataType (&flux)[NCONSRV], DataType vis_0)
{
    DataType primtv[NPRIMTV];
    Consrv2Primtv(u, primtv);

    DataType rhom = 1.0 / u[0];
    DataType drho_dx = u_grad[0];
    DataType du_dx = (u_grad[1] - primtv[1] * drho_dx) * rhom;
    DataType dT_dx =
        (u_grad[2] - u[2] * u_grad[0] * rhom - u[1] * du_dx) * rhom;
    // DataType dprimtv[NPRIMTV];
    // dprimtv[0] = u_grad[0]; //drho_dx
    // dprimtv[1] = (u_grad[1] - primtv[1]*dprimtv[0])/u[0]; // du_dx
    // //dprimtv[2] = (GAMMA-1) * (u_grad[2] - 0.5*u_grad[0]*primtv[1]*primtv[1]
    // - u[2]* dprimtv[1]);// dP_dx dprimtv[3] = (u_grad[2] -
    // u[2]*u_grad[0]/u[0] - u[1]*dprimtv[1])/u[0]; // dT_dx

    DataType vis = vis_0; //* (primtv[3]/T_R) * std::sqrt(primtv[3]/T_R) *(T_R +
                          // S_R)/(primtv[3] + S_R);

    flux[0] = 0.0;
    flux[1] = 4.0 / 3.0 * vis * du_dx;
    flux[2] = primtv[1] * flux[1] + vis * GAMMA / Pr * dT_dx;
}

void computeBR2Flux(const DataType (&uL)[NCONSRV],
                    const DataType (&uL_grad)[NCONSRV],
                    const DataType (&uR)[NCONSRV],
                    const DataType (&uR_grad)[NCONSRV],
                    const DataType local_det_jac_L,
                    const DataType local_det_jac_R, DataType (&flux)[NCONSRV],
                    DataType (&globalLift_L)[NSP * NCONSRV],
                    DataType (&globalLift_R)[NSP * NCONSRV],
                    const Constant<DataType, ORDER> &Constant_s, DataType vis_0)
{
    auto Mass_inv = invertMatrix<DataType, ORDER>(Constant_s.getMMatrix());
    auto MEntropy = getMMatrixEntropy<DataType, ORDER>();
    auto DMass = Constant_s.getDMatrix();

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
                                          MEntropy[isp][ORDER] * jump * (-0.5) /
                                          local_det_jac_L;

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
        uL_grad_[ivar] = uL_grad[ivar] - localLift_L[ORDER][ivar];
        uR_grad_[ivar] = uR_grad[ivar] - localLift_R[0][ivar];
    }

    DataType flux_L[NCONSRV];
    DataType flux_R[NCONSRV];

    computeVisFlux(uL, uL_grad_, flux_L, vis_0);
    computeVisFlux(uR, uR_grad_, flux_R, vis_0);

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = DataType(0.5) * (flux_L[ivar] + flux_R[ivar]);
    }
}
