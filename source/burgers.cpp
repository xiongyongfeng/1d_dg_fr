#include "burgers.h"
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
        DataType F_L = 0.5 * uL[ivar] * uL[ivar];
        DataType F_R = 0.5 * uR[ivar] * uR[ivar];
        DataType alpha = std::max(std::abs(uL[ivar]), std::abs(uR[ivar]));
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
        flux[ivar] = 0.5 * u[ivar] * u[ivar];
    }
}

void computeVisFlux(const DataType (&u) [NCONSRV], const DataType (&u_grad)[NCONSRV],
                 DataType (&flux)[NCONSRV], DataType nu)
{
    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = DataType(0.0);
    }
}

void computeBR2Flux(const DataType (&uL) [NCONSRV], 
                    const DataType (&uL_grad)[NCONSRV],
                    const DataType (&uR) [NCONSRV], 
                    const DataType (&uR_grad)[NCONSRV],
                    const DataType local_det_jac_L,
                    const DataType local_det_jac_R,
                    DataType (&flux)[NCONSRV], 
                    DataType (&globalLift_L)[NSP*NCONSRV],
                    DataType (&globalLift_R)[NSP*NCONSRV],
                    DataType nu )
{
    for (int isp = 0; isp < NSP; isp++)
    {
        for (int ivar = 0; ivar < NCONSRV; ivar++)
        {   
            globalLift_L[isp*NCONSRV+ivar] = DataType(0.0);
            globalLift_R[isp*NCONSRV+ivar] = DataType(0.0);
        }
    }

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    { 
        flux[ivar] = DataType(0.0);
    }

}                    
