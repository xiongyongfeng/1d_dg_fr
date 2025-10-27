#include "lad.h"
#include "constants.h"
#include "macro.h"

void Primtv2Consrv(const DataType (&primtv)[NPRIMTV],
                   DataType (&consrv)[NCONSRV])
{
    consrv[0] = primtv[0];
}

void Consrv2Primtv(const DataType (&consrv)[NCONSRV],
                   DataType (&primtv)[NPRIMTV])
{
    primtv[0] = consrv[0];
}

void computeRiemannFlux(const DataType (&uL)[NCONSRV],
                        const DataType (&uR)[NCONSRV],
                        DataType (&flux)[NCONSRV], DataType a)
{
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {

        DataType F_L = a * uL[ivar];
        DataType F_R = a * uR[ivar];
        DataType alpha = std::abs(a);
        flux[ivar] =
            DataType(0.5) *
            (F_L + F_R - alpha * (uR[ivar] - uL[ivar])); // Lax-Friedrichs
    }
}

void computeFlux(const DataType (&u)[NCONSRV], DataType (&flux)[NCONSRV],
                 DataType a)
{
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = a * u[ivar];
    }
}

void computeVisFlux(const DataType (&u)[NCONSRV],
                    const DataType (&u_grad)[NCONSRV],
                    DataType (&flux)[NCONSRV], DataType nu)
{
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = nu * u_grad[ivar];
    }
}

void computeBR2Flux(const DataType (&uL)[NCONSRV],
                    const DataType (&uL_grad)[NCONSRV],
                    const DataType (&uR)[NCONSRV],
                    const DataType (&uR_grad)[NCONSRV],
                    const DataType local_det_jac_L,
                    const DataType local_det_jac_R, DataType (&flux)[NCONSRV],
                    DataType (&globalLift_L)[NSP * NCONSRV],
                    DataType (&globalLift_R)[NSP * NCONSRV],
                    const Constant<DataType, ORDER> &Constant_s, DataType nu)
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
        uL_grad_[ivar] = localLift_L[ORDER][ivar];
        uR_grad_[ivar] = localLift_R[0][ivar];
    }

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = DataType(0.5) * nu * (uL_grad[ivar] - uL_grad_[ivar]) +
                     DataType(0.5) * nu * (uR_grad[ivar] - uR_grad_[ivar]);
    }
}
