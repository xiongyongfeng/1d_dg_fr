#include "flux.h"

DataType LinearAdvectionDiffusionFlux::computeRiemannFlux(DataType uL,
                                                          DataType uR) const
{
    DataType F_L = a * uL;
    DataType F_R = a * uR;
    DataType alpha = std::abs(a);
    return DataType(0.5) * (F_L + F_R - alpha * (uR - uL)); // Lax-Friedrichs
}

DataType BurgersFlux::computeRiemannFlux(DataType uL, DataType uR) const
{
    DataType a_hat = DataType(0.5) * (uL + uR); // Roe平均速度
    DataType F_L = uL * uL / DataType(2.0);
    DataType F_R = uR * uR / DataType(2.0);
    return DataType(0.5) * (F_L + F_R) -
           DataType(0.5) * std::abs(a_hat) * (uR - uL);
}