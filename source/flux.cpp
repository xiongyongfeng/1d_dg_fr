#include "flux.h"

DataType LinearAdvectionDiffusionFlux::computeRiemannFlux(DataType uL,
                                                          DataType uR) const
{
    DataType F_L = a * uL;
    DataType F_R = a * uR;
    DataType alpha = std::abs(a);
    return DataType(0.5) * (F_L + F_R - alpha * (uR - uL)); // Lax-Friedrichs
}

double BurgersFlux::computeRiemannFlux(double uL, double uR) const
{
    double a_hat = 0.5 * (uL + uR); // Roe平均速度
    double F_L = uL * uL / 2.0;
    double F_R = uR * uR / 2.0;
    return 0.5 * (F_L + F_R) - 0.5 * std::abs(a_hat) * (uR - uL);
}