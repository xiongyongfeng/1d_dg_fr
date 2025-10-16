#pragma once
#include <array>
#include <cstddef>
#include <iostream>
#include <cmath>

template <typename T, size_t ORDER>
constexpr auto getLGLPoints() //order = 1,..,5
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
            T(-1),
            T(-std::sqrt(3.0)/std::sqrt(7.0)),
            T(0),
            T(std::sqrt(3.0)/std::sqrt(7.0)),
            T(1)
        };
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
        return std::array<T, ORDER + 1>{T(1.0 / 3.0), T(4.0 / 3.0), T(1.0 / 3.0)};
    }
    else if constexpr (ORDER == 3)
    {
        return std::array<T, ORDER + 1>{
            T(1.0 / 6.0), T(5.0 / 6.0), T(5.0 / 6.0), T(1.0 / 6.0)};
    }
    else if constexpr (ORDER == 4)
    {
        return std::array<T, ORDER + 1>{
            T(1.0/10.0),
            T(49.0/90.0),
            T(32.0/45.0),
            T(49.0/90.0),
            T(1.0/10.0)};
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
    /*
    M1 = Matrix([[2/3, 1/3],[1/3, 2/3]])
M2 = Matrix([[4/15, 2/15, -1/15],[2/15, 16/15, 2/15],[-1/15, 2/15, 4/15]])
M3 = Matrix([
    [0.14285714,  0.05323971, -0.05323971,  0.02380952],
    [0.05323971,  0.71428572,  0.11904763, -0.05323971],
    [-0.05323971, 0.11904763,  0.71428572,  0.05323971],
    [0.02380952, -0.05323971,  0.05323971,  0.14285714]
])
M4 = Matrix([[ 0.08888889,  0.02592593, -0.02962963 , 0.02592593 ,-0.01111111],
 [ 0.02592593 , 0.48395062 , 0.0691358 , -0.06049383 , 0.02592593],
 [-0.02962963 , 0.0691358 ,  0.63209877 , 0.0691358 , -0.02962963],
 [ 0.02592593, -0.06049383 , 0.0691358  , 0.48395062 , 0.02592593],
 [-0.01111111 , 0.02592593,-0.02962963 , 0.02592593 , 0.08888889]])
    */
    static_assert(ORDER <= 4, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{T(2.0) / T(3.0), T(1.0) / T(3.0)},
                          ArrayType{T(1.0) / T(3.0), T(2.0) / T(3.0)}};
    }
    else if constexpr (ORDER == 2)
    { 
        return MatrixType{ArrayType{T(4.0) / T(15.0), T(2.0) / T(15.0), T(-1.0) / T(15.0)},
                          ArrayType{T(2.0) / T(15.0), T(16.0) / T(15.0), T(2.0) / T(15.0)},
                          ArrayType{T(-1.0) / T(15.0), T(2.0) / T(15.0), T(4.0) / T(15.0)}};                    
    } 
    else if constexpr (ORDER == 3)
    {
        return MatrixType{
            ArrayType{T(1.0) / T(7.0), T(3.0) / T(56.0), T(-3.0) / T(56.0), T(1.0) / T(42.0)},
            ArrayType{T(3.0) / T(56.0), T(5.0) / T(7.0), T(1.0) / T(8.0), T(-3.0) / T(56.0)},
            ArrayType{T(-3.0) / T(56.0), T(1.0) / T(8.0), T(5.0) / T(7.0), T(3.0) / T(56.0)},
            ArrayType{T(1.0) / T(42.0), T(-3.0) / T(56.0), T(3.0) / T(56.0), T(1.0) / T(7.0)}};
    }
    else if constexpr (ORDER == 4)
    {
        return MatrixType{
            ArrayType{T(8.0) / T(90.0), T(7.0) / T(270.0), T(-8.0) / T(270.0), T(7.0) / T(270.0), T(-4.0) / T(360.0)},
            ArrayType{T(7.0) / T(270.0), T(1096.0) / T(2250.0), T(1568.0) / T(22500.0), T(-1369.0) / T(22500.0), T(7.0) / T(270.0)},
            ArrayType{T(-8.0) / T(270.0), T(1568.0) / T(22500.0), T(1412.0) / T(1755.0), T(1568.0) / T(22500.0), 	T(-8.0) / T(270.0)},
            ArrayType{T(7.0) / T(270.0),	T(-1369.0) / T(22500.0), T(1568.0) / T(22500.0), T(1096.0) / T(2250.0), T(7.0) / T(270.0)},
            ArrayType{T(-4.0) / T(360.0), T(7.0) / T(270.0), T(-8.0) / T(270.0), T(7.0) / T(270.0), T(8.0) / T(90.0)}};
    }
    else {
        return std::array<std::array<T, 0>, 0>{};
    }

            
   
}

// entropy mass matrix
// M_ij = \int_{-1}^{1} l_i l_j d\xi
template <typename T, size_t ORDER>
constexpr auto getMMatrixEntropy()
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
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
constexpr auto getKMatrix()
{
    static_assert(ORDER <= 1, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{T(0.5), -T(0.5)},
                          ArrayType{-T(0.5), T(0.5)}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};

// Stiffness Matrix,
// K_ij = \int_{-1}^1 \partial l_i / \partial \xi \partial l_j / \partial \xi d\xi = M_in*D_nj
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

template <typename T, size_t ORDER>
constexpr auto getTMatrix(T a)
{
    static_assert(ORDER <= 3, "Unsupported ORDER");
    using ArrayType = std::array<T, ORDER + 1>;
    using MatrixType = std::array<std::array<T, ORDER + 1>, ORDER + 1>;
    if constexpr (ORDER == 1)
    {
        return MatrixType{ArrayType{-T(0), T(1)},
                          ArrayType{-T(a), T(1+a)}};
    }
    else if constexpr (ORDER == 2)
    {
        return MatrixType{ArrayType{T(0), T(0), T(1)},
                          ArrayType{T(a*(1.0+a)/2.0), -T(a*(2.0+a)), T((1.0+a)*(2.0+a)/2.0)},
                          ArrayType{T(a*(1.0+2.0*a)), -T(4.0*a*(1+a)), T((1.0+a)*(1.0+2.0*a))}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};
