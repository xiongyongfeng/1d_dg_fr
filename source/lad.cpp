#include "lad.h"
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