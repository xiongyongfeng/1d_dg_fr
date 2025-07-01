#pragma once
#include "macro.h"
#include <iostream>
class Flux
{
  public:
    virtual DataType computeRiemannFlux(DataType uL, DataType uR) const = 0;
    virtual DataType computeFlux(DataType u) const = 0;
    virtual ~Flux() = default;
};

// 线性对流扩散通量
class LinearAdvectionDiffusionFlux : public Flux
{
    DataType a;

  public:
    LinearAdvectionDiffusionFlux(DataType a) : a(a) {}
    DataType computeRiemannFlux(DataType uL, DataType uR) const override;
    DataType computeFlux(DataType u) const override { return a * u; }
};

// Burgers方程通量（Roe格式）
class BurgersFlux : public Flux
{
  public:
    double computeRiemannFlux(double uL, double uR) const override;

    DataType computeFlux(DataType u) const override
    {
        return DataType(0.5) * u * u;
    }
};