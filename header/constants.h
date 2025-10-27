#pragma once
#include "GaussianQuadrature.h"
#include "LGLDefines.h"
#include "MatrixUtil.h"
#include "macro.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>

template <typename T, size_t ORDER>
class Constant
{
    using BasisType = std::function<T(const T)>;
    using BasisList = std::array<BasisType, ORDER + 1>;
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;

  public:
    LGLDefines<T, ORDER> lgl_def;
    ArrayType lgl_points;
    ArrayType lgl_weights;

    BasisList BasisFuncs;
    BasisList DbasisFuncs;

  protected:
    MatrixType MMatrix;
    MatrixType SMatrix;
    MatrixType TMatrix;
    MatrixType DMatrix;

  private:
    void setFuncs()
    {
        using namespace std::placeholders;
        auto Funcs = LGLDefines<T, ORDER>();
        for (size_t i = 0; i < ORDER + 1; i++)
        {
            BasisFuncs[i] = std::bind(&LGLDefines<T, ORDER>::Li, Funcs, i, _1);
            DbasisFuncs[i] =
                std::bind(&LGLDefines<T, ORDER>::Li_deriv, Funcs, i, _1);
        }
    }
    void setMMatrix()
    {
        // compute MMatrix_{ij} = \int_{-1}^{1} l_i(x) l_j(x) dx
        const int nq =
            ORDER +
            1; // number of quadrature points (<=5 for our GaussianQuadrature)
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
            {
                auto f = [this, i, j](T x)
                { return BasisFuncs[i](x) * BasisFuncs[j](x); };
                MMatrix[i][j] = GaussianQuadrature::integrate(f, nq);
            }
        }
    }
    void setSmatrix()
    {
        // K_ij = \int_{-1}^1 \partial l_j / \partial \xi \partial l_i /
        // \partial \xi
        // d\xi = M_in*D_nj
        const int nq =
            ORDER +
            1; // number of quadrature points (<=5 for our GaussianQuadrature)
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
            {
                auto f = [this, i, j](T x)
                { return DbasisFuncs[j](x) * BasisFuncs[i](x); };
                SMatrix[i][j] = GaussianQuadrature::integrate(f, nq);
            }
        }
    }
    void setDmatrix()
    {
        // compute D_{ij} =  \partial l_j / \partial \xi at \xi_i
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
            {
                DMatrix[i][j] = DbasisFuncs[j](lgl_points[i]);
            }
        }
    }

    void setConstants()
    {
        setMMatrix();
        setDmatrix();
        setSmatrix();
        valueTMatrix();
    }

  public:
    Constant<T, ORDER>() : lgl_def()
    {
        lgl_points = lgl_def.LGLPoints;
        lgl_weights = lgl_def.LGLWeights;
        // initialize basis function wrappers and constant matrices
        setFuncs();
        setConstants();
    }
    auto getLGLPoints() const { return lgl_points; }
    auto getLGLWeights() const { return lgl_weights; }

    const auto getLGLBasis() const { return BasisFuncs; }
    const auto getLGLBasis_deriv() const { return DbasisFuncs; }
    const auto getMMatrix() const { return MMatrix; }
    const auto getSMatrix() const { return SMatrix; }
    const auto getDMatrix() const { return DMatrix; }
    const auto getTMatrix() const { return TMatrix; }

    void valueTMatrix(T a = 1.0)
    {

        for (int i = 0; i < ORDER + 1; i++)
        {
            for (int j = 0; j < ORDER + 1; j++)
            {
                TMatrix[j][i] = BasisFuncs[i](1 + a + a * lgl_points[j]);
            }
        }
    };

    // Print all stored members for debugging/inspection
    void show(std::ostream &os = std::cout) const
    {
        os << std::scientific << std::setprecision(12);
        os << "LGL points: ";
        for (size_t i = 0; i < ORDER + 1; ++i)
            os << lgl_points[i] << (i + 1 < ORDER + 1 ? ", " : "\n");

        os << "LGL weights: ";
        for (size_t i = 0; i < ORDER + 1; ++i)
            os << lgl_weights[i] << (i + 1 < ORDER + 1 ? ", " : "\n");

        os << "\nMMatrix:\n";
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
                os << MMatrix[i][j] << (j + 1 < ORDER + 1 ? ", " : "");
            os << "\n";
        }

        os << "\nSMatrix:\n";
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
                os << SMatrix[i][j] << (j + 1 < ORDER + 1 ? ", " : "");
            os << "\n";
        }

        os << "\nDMatrix (D_{ij} = d l_j / d\\xi at xi):\n";
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
                os << DMatrix[i][j] << (j + 1 < ORDER + 1 ? ", " : "");
            os << "\n";
        }

        os << "\nTMatrix:\n";
        for (size_t i = 0; i < ORDER + 1; ++i)
        {
            for (size_t j = 0; j < ORDER + 1; ++j)
                os << TMatrix[i][j] << (j + 1 < ORDER + 1 ? ", " : "");
            os << "\n";
        }

        // Evaluate basis functions and their derivatives at LGL points
        os << "\nBasis functions evaluated at LGL points:\n";
        for (size_t b = 0; b < ORDER + 1; ++b)
        {
            os << "phi_" << b << ": ";
            for (size_t p = 0; p < ORDER + 1; ++p)
            {
                T val = BasisFuncs[b](lgl_points[p]);
                os << val << (p + 1 < ORDER + 1 ? ", " : "\n");
            }
        }

        os << "\nBasis derivatives at LGL points:\n";
        for (size_t b = 0; b < ORDER + 1; ++b)
        {
            os << "phi_" << b << "' : ";
            for (size_t p = 0; p < ORDER + 1; ++p)
            {
                T val = DbasisFuncs[b](lgl_points[p]);
                os << val << (p + 1 < ORDER + 1 ? ", " : "\n");
            }
        }

        os << std::defaultfloat;
    }

    void show() {}
};

// template <typename T, size_t ORDER>
// const auto getLGLBasis()
// {
//     using BasisType = std::function<T(const T)>;
//     using BasisList = std::array<BasisType, ORDER + 1>;
//     using ArrayType = std::array<T, ORDER + 1>;
//     auto Funcs = LGLDefines<T, ORDER>();
//     auto LGLPoints = Funcs.LGLPoints;
//     BasisList LGLBasis;
//     for (size_t i = 0; i < ORDER + 1; i++)
//     {
//         using namespace std::placeholders;
//         LGLBasis[i] = std::bind(&LGLDefines<T, ORDER>::Li, Funcs, i, _1);
//     }
//     return LGLBasis;
// }
// template <typename T, size_t ORDER>
// const auto getLGLBasis_deriv()
// {
//     using BasisType = std::function<T(const T)>;
//     using BasisList = std::array<BasisType, ORDER + 1>;
//     using ArrayType = std::array<T, ORDER + 1>;
//     auto Funcs = LGLDefines<T, ORDER>();
//     auto LGLPoints = Funcs.LGLPoints;
//     BasisList LGLBasis_deriv;
//     for (size_t i = 0; i < ORDER + 1; i++)
//     {
//         using namespace std::placeholders;
//         LGLBasis_deriv[i] =
//             std::bind(&LGLDefines<T, ORDER>::Li_deriv, Funcs, i, _1);
//     }
//     return LGLBasis_deriv;
// }

// // mass matrix
// // M_ij = \int_{-1}^{1} l_i l_j d\xi
// template <typename T, size_t ORDER>
// constexpr auto getMMatrix()
// {
//     static_assert(ORDER <= 4, "Unsupported ORDER");
//     using ArrayType = std::array<T, ORDER + 1>;
//     using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
//     if constexpr (ORDER == 1)
//     {
//         return MatrixType{ArrayType{T(2.0) / T(3.0), T(1.0) / T(3.0)},
//                           ArrayType{T(1.0) / T(3.0), T(2.0) / T(3.0)}};
//     }
//     else if constexpr (ORDER == 2)
//     {
//         return MatrixType{
//             ArrayType{T(4.0) / T(15.0), T(2.0) / T(15.0), T(-1.0) / T(15.0)},
//             ArrayType{T(2.0) / T(15.0), T(16.0) / T(15.0), T(2.0) / T(15.0)},
//             ArrayType{T(-1.0) / T(15.0), T(2.0) / T(15.0), T(4.0) /
//             T(15.0)}};
//     }
//     else if constexpr (ORDER == 3)
//     {
//         return MatrixType{ArrayType{0.1428571426176, 0.0532397107162,
//                                     -0.0532397149088, 0.0238095240491},
//                           ArrayType{0.0532397107162, 0.7142857154836,
//                                     0.1190476262350, -0.0532397149088},
//                           ArrayType{-0.0532397149088, 0.1190476262350,
//                                     0.7142857154836, 0.0532397107162},
//                           ArrayType{0.0238095240491, -0.0532397149088,
//                                     0.0532397107162, 0.1428571426176}};
//     }
//     else if constexpr (ORDER == 4)
//     {
//         return MatrixType{
//             ArrayType{T(8.0) / T(90.0), T(7.0) / T(270.0), T(-8.0) /
//             T(270.0),
//                       T(7.0) / T(270.0), T(-4.0) / T(360.0)},
//             ArrayType{T(7.0) / T(270.0), T(1096.0) / T(2250.0),
//                       T(1568.0) / T(22500.0), T(-1369.0) / T(22500.0),
//                       T(7.0) / T(270.0)},
//             ArrayType{T(-8.0) / T(270.0), T(1568.0) / T(22500.0),
//                       T(1412.0) / T(1755.0), T(1568.0) / T(22500.0),
//                       T(-8.0) / T(270.0)},
//             ArrayType{T(7.0) / T(270.0), T(-1369.0) / T(22500.0),
//                       T(1568.0) / T(22500.0), T(1096.0) / T(2250.0),
//                       T(7.0) / T(270.0)},
//             ArrayType{T(-4.0) / T(360.0), T(7.0) / T(270.0), T(-8.0) /
//             T(270.0),
//                       T(7.0) / T(270.0), T(8.0) / T(90.0)}};
//     }
//     else
//     {
//         return std::array<std::array<T, 0>, 0>{};
//     }
// }

// entropy mass matrix
// M_ij = \int_{-1}^{1} l_i l_j d\xi
template <typename T, size_t ORDER>
constexpr auto getMMatrixEntropy()
{
    static_assert(ORDER <= 2, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{T(1.0), T(0.0)}, ArrayType{T(0.0), T(1.0)}};
    }
    else if constexpr (ORDER == 2)
    {
        return MatrixType{ArrayType{0.33333333, 0.0, 0.0},
                          ArrayType{0.0, 1.33333333, 0.0},
                          ArrayType{0.0, 0.0, 0.33333333}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};

template <typename T, size_t ORDER>
constexpr auto
invertMatrix(const std::array<std::array<T, ORDER + 1>, ORDER + 1> &mat)
{
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    MatrixType invMat{};
    MatrixUtil<ORDER + 1, DataType>::inverseMatrixLU(mat, invMat);
    return invMat;
}

// // Difference Matrix
// // D_{ij} =  \partial l_j / \partial \xi at \xi_i
// template <typename T, size_t ORDER>
// constexpr auto getDMatrix()
// {
//     static_assert(ORDER <= 3, "Unsupported ORDER");
//     using ArrayType = std::array<T, ORDER + 1>;
//     using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
//     if constexpr (ORDER == 1)
//     {
//         return MatrixType{ArrayType{-T(0.5), T(0.5)},
//                           ArrayType{-T(0.5), T(0.5)}};
//     }
//     else if constexpr (ORDER == 2)
//     {
//         return MatrixType{ArrayType{-T(1.50), T(2.00), -T(0.50)},
//                           ArrayType{-T(0.5), T(0), T(0.5)},
//                           ArrayType{T(0.5), -T(2), T(1.5)}};
//     }
//     else
//     {
//         return std::array<std::array<T, 0>, 0>{};
//     }
// };

// // Stiffness Matrix,
// // K_ij = \int_{-1}^1 \partial l_i / \partial \xi \partial l_j / \partial \xi
// // d\xi = M_in*D_nj
// template <typename T, size_t ORDER>
// constexpr auto getSMatrix()
// {
//     static_assert(ORDER <= 5, "Unsupported ORDER");
//     using ArrayType = std::array<T, ORDER + 1>;
//     using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
//     if constexpr (ORDER == 1)
//     {

//         return MatrixType{ArrayType{-T(0.5), T(0.5)},
//                           ArrayType{-T(0.5), T(0.5)}};
//     }
//     else if constexpr (ORDER == 2)
//     {
//         return MatrixType{
//             ArrayType{-5.00000000e-01, 6.66666667e-01, -1.66666667e-01},
//             ArrayType{-6.66666667e-01, -5.55111512e-17, 6.66666667e-01},
//             ArrayType{1.66666667e-01, -6.66666667e-01, 5.00000000e-01}};
//     }
//     else if constexpr (ORDER == 3)
//     {
//         return MatrixType{ArrayType{-0.5000000000000, 0.6741808286458,
//                                     -0.2575141619791, 0.0833333333333},
//                           ArrayType{-0.6741808286458, -0.0000000000000,
//                                     0.9316949906249, -0.2575141619791},
//                           ArrayType{0.2575141619791, -0.9316949906249,
//                                     0.0000000000000, 0.6741808286458},
//                           ArrayType{-0.0833333333333, 0.2575141619791,
//                                     -0.6741808286458, 0.5000000000000}};
//     }
//     else if constexpr (ORDER == 4)
//     {
//         return MatrixType{
//             ArrayType{-0.5000000000000, 0.6756502488724, -0.2666666666667,
//                       0.1410164177942, -0.0500000000000},
//             ArrayType{-0.6756502488724, -0.0000000000000, 0.9504601441390,
//                       -0.4158263130608, 0.1410164177942},
//             ArrayType{0.2666666666667, -0.9504601441390, 0.0000000000000,
//                       0.9504601441390, -0.2666666666667},
//             ArrayType{-0.1410164177942, 0.4158263130608, -0.9504601441390,
//                       0.0000000000000, 0.6756502488724},
//             ArrayType{0.0500000000000, -0.1410164177942, 0.2666666666667,
//                       -0.6756502488724, 0.5000000000000}};
//     }
//     else if constexpr (ORDER == 5)
//     {
//         return MatrixType{
//             ArrayType{-0.5000000000000, 0.6760943957546, -0.2690791513537,
//                       0.1496456432117, -0.0899942209460, 0.0333333333333},
//             ArrayType{-0.6760943957546, -0.0000000000000, 0.9550538393084,
//                       -0.4363165869208, 0.2473513643131, -0.0899942209460},
//             ArrayType{0.2690791513537, -0.9550538393084, -0.0000000000000,
//                       0.9726456316638, -0.4363165869208, 0.1496456432117},
//             ArrayType{-0.1496456432117, 0.4363165869208, -0.9726456316638,
//                       0.0000000000000, 0.9550538393084, -0.2690791513537},
//             ArrayType{0.0899942209460, -0.2473513643131, 0.4363165869208,
//                       -0.9550538393084, 0.0000000000000, 0.6760943957546},
//             ArrayType{-0.0333333333333, 0.0899942209460, -0.1496456432117,
//                       0.2690791513537, -0.6760943957546, 0.5000000000000}};
//     }
//     else
//     {
//         return std::array<std::array<T, 0>, 0>{};
//     }
// };

// template <typename T, size_t ORDER>
// auto getTMatrix(T a)
// {
//     static_assert(ORDER <= 5, "Unsupported ORDER");
//     using ArrayType = std::array<T, ORDER + 1>;
//     using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
//     using BasisType = std::function<T(const T)>;
//     using BasisList = std::array<BasisType, ORDER + 1>;

//     MatrixType Tmat;
//     BasisList Basis = getLGLBasis<T, ORDER>();
//     ArrayType Points = getLGLPoints<T, ORDER>();

//     for (int i = 0; i < ORDER + 1; i++)
//     {
//         for (int j = 0; j < ORDER + 1; j++)
//         {
//             Tmat[j][i] = Basis[i](1 + a + a * Points[j]);
//         }
//     }
//     return Tmat;
// };
