#include "ns.h"
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
    DataType lambda_l = std::fabs(primtv_l[1]) + GetSoundSpeed(uL, primtv_l);
    DataType lambda_r = std::fabs(primtv_r[1]) + GetSoundSpeed(uR, primtv_r);
    DataType lambda = std::max(lambda_l, lambda_r);

    DataType flux_l[NCONSRV];
    DataType flux_r[NCONSRV];
    computeFlux(uL, flux_l);
    computeFlux(uR, flux_r);

    for (int ivar = 0; ivar < NCONSRV; ivar++)
    {
        flux[ivar] = 0.5 * (flux_l[ivar] + flux_r[ivar]) -
                     0.5 * lambda * (uR[ivar] - uL[ivar]);
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