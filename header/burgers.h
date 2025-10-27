#include "constants.h"
#include "macro.h"
void Primtv2Consrv(const DataType (&primtv)[NPRIMTV],
                   DataType (&consrv)[NCONSRV]);
void Consrv2Primtv(const DataType (&consrv)[NCONSRV],
                   DataType (&primtv)[NPRIMTV]);
void computeRiemannFlux(const DataType (&uL)[NCONSRV],
                        const DataType (&uR)[NCONSRV],
                        DataType (&flux)[NCONSRV], DataType a = 0);

void computeFlux(const DataType (&u)[NCONSRV], DataType (&flux)[NCONSRV],
                 DataType a = 0);
void computeVisFlux(const DataType (&u)[NCONSRV],
                    const DataType (&u_grad)[NCONSRV],
                    DataType (&flux)[NCONSRV], DataType nu = 0);

void computeBR2Flux(
    const DataType (&uL)[NCONSRV], const DataType (&uL_grad)[NCONSRV],
    const DataType (&uR)[NCONSRV], const DataType (&uR_grad)[NCONSRV],
    const DataType local_det_jac_L, const DataType local_det_jac_R,
    DataType (&flux)[NCONSRV], DataType (&globalLift_L)[NSP * NCONSRV],
    DataType (&globalLift_R)[NSP * NCONSRV],
    const Constant<DataType, ORDER> &Constant_s, DataType nu = 0);