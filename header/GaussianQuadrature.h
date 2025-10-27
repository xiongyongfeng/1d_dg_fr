#pragma once
#include "macro.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

class GaussianQuadrature
{
    using FuncType = std::function<DataType(const DataType)>;

  private:
    // 获取高斯-勒让德求积的节点和权重
    static void gauss_legendre_points(int n, std::vector<double> &nodes,
                                      std::vector<double> &weights)
    {
        nodes.resize(n);
        weights.resize(n);

        switch (n)
        {
        // 预定义的低阶高斯点（更高效）
        case 1:
            nodes[0] = 0.0;
            weights[0] = 2.0;
            break;
        case 2:
            nodes[0] = -0.5773502691896257;
            nodes[1] = 0.5773502691896257;
            weights[0] = 1.0;
            weights[1] = 1.0;
            break;
        case 3:
            nodes[0] = -0.7745966692414834;
            nodes[1] = 0.0;
            nodes[2] = 0.7745966692414834;
            weights[0] = 0.5555555555555556;
            weights[1] = 0.8888888888888888;
            weights[2] = 0.5555555555555556;
            break;
        case 4:
            nodes[0] = -0.8611363115940526;
            nodes[1] = -0.3399810435848563;
            nodes[2] = 0.3399810435848563;
            nodes[3] = 0.8611363115940526;
            weights[0] = 0.3478548451374538;
            weights[1] = 0.6521451548625461;
            weights[2] = 0.6521451548625461;
            weights[3] = 0.3478548451374538;
            break;
        case 5:
            nodes[0] = -0.9061798459386640;
            nodes[1] = -0.5384693101056831;
            nodes[2] = 0.0;
            nodes[3] = 0.5384693101056831;
            nodes[4] = 0.9061798459386640;
            weights[0] = 0.2369268850561891;
            weights[1] = 0.4786286704993665;
            weights[2] = 0.5688888888888889;
            weights[3] = 0.4786286704993665;
            weights[4] = 0.2369268850561891;
            break;
        default:
            // 对于更高阶，待补充
            break;
        }
    }

  public:
    // N点高斯积分
    static DataType integrate(FuncType f, int n, DataType a = -1,
                              DataType b = 1)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("Number of points must be positive");
        }

        std::vector<double> nodes, weights;
        gauss_legendre_points(n, nodes, weights);

        DataType sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            // 坐标变换从 [-1, 1] 到 [a, b]
            DataType x = (b - a) / 2.0 * nodes[i] + (a + b) / 2.0;
            sum += weights[i] * f(x);
        }

        return sum * (b - a) / 2.0;
    }
};