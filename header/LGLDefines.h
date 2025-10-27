#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>

template <typename T, size_t ORDER>
constexpr auto getLGLPoints() // order = 1,..,5
{
    static_assert(ORDER <= 4, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return std::array<T, ORDER + 1>{T(-1), T(1)};
    }
    else if constexpr (ORDER == 2)
    {
        return std::array<T, ORDER + 1>{T(-1), T(0), T(1)};
    }
    else if constexpr (ORDER == 3)
    {
        return std::array<T, ORDER + 1>{T(-1), T(-std::sqrt(1.0 / 5.0)),
                                        T(std::sqrt(1.0 / 5.0)), T(1)};
    }
    else if constexpr (ORDER == 4)
    {
        return std::array<T, ORDER + 1>{
            T(-1), T(-std::sqrt(3.0) / std::sqrt(7.0)), T(0),
            T(std::sqrt(3.0) / std::sqrt(7.0)), T(1)};
    }
    else
    {
        return std::array<T, 0>{};
    }
}

template <typename T, size_t ORDER>
constexpr auto getLGLWeights() // order = 1,..,5
{
    static_assert(ORDER <= 4, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return std::array<T, ORDER + 1>{T(1), T(1)};
    }
    else if constexpr (ORDER == 2)
    {
        return std::array<T, ORDER + 1>{T(1.0 / 3.0), T(4.0 / 3.0),
                                        T(1.0 / 3.0)};
    }
    else if constexpr (ORDER == 3)
    {
        return std::array<T, ORDER + 1>{T(1.0 / 6.0), T(5.0 / 6.0),
                                        T(5.0 / 6.0), T(1.0 / 6.0)};
    }
    else if constexpr (ORDER == 4)
    {
        return std::array<T, ORDER + 1>{T(1.0 / 10.0), T(49.0 / 90.0),
                                        T(32.0 / 45.0), T(49.0 / 90.0),
                                        T(1.0 / 10.0)};
    }
    else
    {
        return std::array<T, 0>{};
    }
}

template <typename T, size_t ORDER>
class LGLDefines
{
    using BasisType = std::function<T(const T)>;
    using BasisList = std::array<BasisType, ORDER + 1>;
    using ArrayType = std::array<T, ORDER + 1>;

  public:
    ArrayType LGLPoints;
    ArrayType LGLWeights;

  public:
    LGLDefines<T, ORDER>()
    {
        LGLPoints = getLGLPoints<T, ORDER>();
        LGLWeights = getLGLWeights<T, ORDER>();
    }

    T Li(size_t i, T xi)
    {
        T rt = T(1.0);
        T denominator = T(1.0);
        for (size_t k = 0; k < ORDER + 1; k++)
        {
            if (i != k)
            {
                rt *= xi - LGLPoints[k];
                denominator *= LGLPoints[i] - LGLPoints[k];
            }
        }
        rt = rt / denominator;
        return rt;
    }

    T Li_deriv(size_t i, T xi)
    {

        T denominator = T(1.0);
        for (size_t k = 0; k < ORDER + 1; k++)
        {
            if (i != k)
            {
                denominator *= LGLPoints[i] - LGLPoints[k];
            }
        }

        T rt = T(0.0);
        for (size_t j = 0; j < ORDER + 1; j++)
        {
            T numerator = T(1.0);
            for (size_t k = 0; k < ORDER + 1; k++)
            {
                if (i != k && j != k && i != j)
                {
                    numerator *= xi - LGLPoints[k];
                }
            }
            if (j != i)
            {
                rt += numerator;
            }
        }

        rt /= denominator;
        return rt;
    }
};