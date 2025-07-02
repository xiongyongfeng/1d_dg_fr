#pragma once
#include <array>
#include <cstddef>
#include <iostream>

template <typename T, size_t ORDER>
constexpr auto getLGLPoints()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return std::array<T, ORDER + 1>{T(-1), T(1)};
    }
    else if constexpr (ORDER == 2)
    {
        return std::array<T, ORDER + 1>{T(-1), T(0), T(1)};
    }
    else
    {
        return std::array<T, 0>{};
    }
}

template <typename T, size_t ORDER>
constexpr auto getLGLWeights()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return std::array<T, ORDER + 1>{T(1), T(1)};
    }
    else if constexpr (ORDER == 2)
    {
        return std::array<T, ORDER + 1>{
            T(1.0) / T(3.0), T(1.3333333333333333334), T(1.0) / T(3.0)};
    }
    else
    {
        return std::array<T, 0>{};
    }
}

// mass matrix
// M_ij = \int_{-1}^{1} l_i l_j d\xi
template <typename T, size_t ORDER>
constexpr auto getMMatrix()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{T(2.0) / T(3.0), T(1.0) / T(3.0)},
                          ArrayType{T(1.0) / T(3.0), T(2.0) / T(3.0)}};
    }
    else if constexpr (ORDER == 2)
    {
        return MatrixType{ArrayType{0.26666667, 0.13333333, -0.06666667},
                          ArrayType{0.13333333, 1.06666667, 0.13333333},
                          ArrayType{-0.06666667, 0.13333333, 0.26666667}};
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
    static_assert(ORDER <= 3 && ORDER >= 1, "Unsupported ORDER");
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    MatrixType invMat{};

    if constexpr (ORDER == 1)
    {
        // --- 1阶矩阵求逆公式 ---
        T det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
        if (det == 0)
            throw std::invalid_argument("Matrix is singular");

        T invDet = T(1) / det;
        invMat = {{{mat[1][1] * invDet, -mat[0][1] * invDet},
                   {-mat[1][0] * invDet, mat[0][0] * invDet}}};
    }
    else if constexpr (ORDER == 2)
    {
        // --- 2阶矩阵求逆公式 ---
        T det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
                mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
                mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
        if (det == 0)
            throw std::invalid_argument("Matrix is singular");

        T invDet = T(1) / det;
        invMat = {{{(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) * invDet,
                    (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * invDet,
                    (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * invDet},
                   {(mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * invDet,
                    (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * invDet,
                    (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * invDet},
                   {(mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * invDet,
                    (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * invDet,
                    (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * invDet}}};
    }
    return invMat;
}

// Difference Matrix
// D_{ij} =  \partial l_j / \partial \xi at \xi_i
template <typename T, size_t ORDER>
constexpr auto getDMatrix()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{-T(0.5), T(0.5)},
                          ArrayType{-T(0.5), T(0.5)}};
    }
    else if constexpr (ORDER == 2)
    {
        return MatrixType{ArrayType{-T(1.50), T(2.00), -T(0.50)},
                          ArrayType{-T(0.5), T(0), T(0.5)},
                          ArrayType{T(0.5), -T(2), T(1.5)}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};

// S Matrix, S = MD
// S_ij = \int_{-1}^1 l_i \partial l_j / \partial \xi d\xi = M_in*D_nj
template <typename T, size_t ORDER>
constexpr auto getSMatrix()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {

        return MatrixType{ArrayType{-T(0.5), T(0.5)},
                          ArrayType{-T(0.5), T(0.5)}};
    }
    else if constexpr (ORDER == 2)
    {
        return MatrixType{
            ArrayType{-5.00000000e-01, 6.66666667e-01, -1.66666667e-01},
            ArrayType{-6.66666667e-01, -5.55111512e-17, 6.66666667e-01},
            ArrayType{1.66666667e-01, -6.66666667e-01, 5.00000000e-01}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};
