#include "macro.h"
#include <cmath>


#ifdef LAD
using ConsrvType = std::array<DataType, NCONSRV>;
inline ConsrvType exact_lad(DataType x, DataType t){
    ConsrvType consrv;

                    // if (geom_pool[iele].x[isp] < DataType(0.25))
                    // {
                    //     elem.u_consrv[isp][ivar] = DataType(10.0);
                    // }
                    // else if (geom_pool[iele].x[isp] > DataType(0.75))
                    // {
                    //     elem.u_consrv[isp][ivar] = DataType(10.0);
                    // }
                    // else
                    // {
                    //     elem.u_consrv[isp][ivar] = DataType(11.0);
                    // }
    consrv[0] = std::sin(2*acos(-1.0)*(x-t));
    return consrv;
}
#endif

#ifdef NS
using ConsrvType = std::array<DataType, NCONSRV>;
inline ConsrvType u0_sod(DataType x){
    ConsrvType consrv;
    if ( x < DataType(0.0) ){
        consrv[0] = 1.0;
        consrv[1] = 0;
        consrv[2] = 2.5;
    }
    else{
        consrv[0] = 0.125;
        consrv[1] = 0;
        consrv[2] = 0.25;
    }
    return consrv;
}
#endif


#ifdef BURGERS
using ConsrvType = std::array<DataType, NCONSRV>;
inline ConsrvType u0_burgers(DataType x){
    ConsrvType consrv;
    consrv[0] = 0.5 + std::sin(x);
    return consrv;
}
#endif
