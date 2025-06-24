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
        return std::array<T, ORDER + 1>{T(-1), T(0), T(1)};
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
        return MatrixType{ArrayType{0, 1, 2}, ArrayType{3, 4, 5},
                          ArrayType{6, 7, 8}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};

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
        return MatrixType{ArrayType{0, 1, 2}, ArrayType{3, 4, 5},
                          ArrayType{6, 7, 8}};
    }
    else
    {
        return std::array<std::array<T, 0>, 0>{};
    }
};
