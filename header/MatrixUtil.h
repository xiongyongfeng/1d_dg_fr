#pragma once
#include "macro.h"
#include <cmath>
#include <cstring>

template <int N, typename DataType>
class MatrixUtil
{
    using ArrayType = std::array<DataType, N>;
    using MatrixType = std::array<std::array<DataType, N>, N>;

  public:
    static void copy(const MatrixType &A, MatrixType &B)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                B[i][j] = A[i][j];
            }
        }
    }
    static void setzero(MatrixType &A)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i][j] = DataType(0.0);
            }
        }
    }

    static void inverseMatrixLU(const MatrixType &A, MatrixType &B)
    {

        MatrixType T, L, U, Q;
        ArrayType y;
        copy(A, T);

        // QA = LU
        LUDecomposition(T, L, U, Q);

        // step2: B = inv(A)
        for (int k = 0; k < N; ++k)
        {
            // forward iteration
            y[0] = Q[0][k];
            for (int i = 1; i < N; ++i)
            {
                y[i] = Q[i][k];
                for (int t = 0; t <= i - 1; ++t)
                {
                    y[i] -= L[i][t] * y[t];
                }
            }

            // backword iteration
            B[N - 1][k] = y[N - 1] / U[N - 1][N - 1];
            for (int i = N - 2; i > -1; --i)
            {
                B[i][k] = y[i];
                for (int t = i + 1; t <= N - 1; ++t)
                {
                    B[i][k] -= U[i][t] * B[t][k];
                }
                B[i][k] /= U[i][i];
            }
        }
    }

    static void LUDecomposition(MatrixType &A, MatrixType &L, MatrixType &U,
                                MatrixType &Q)
    {
        setzero(L);
        setzero(U);
        setzero(Q);

        for (int i = 0; i < N; ++i)
        {
            Q[i][i] = DataType(1.0);
        }

        DataType s[N];

        for (int k = 0; k < N; ++k)
        {
            for (int i = k; i < N; ++i)
            {
                s[i] = A[i][k];
                for (int t = 0; t <= k - 1; ++t)
                {
                    s[i] -= L[i][t] * U[t][k];
                }
            }

            DataType temp = fabs(s[k]);
            int ik = k;
            for (int i = k + 1; i < N; ++i)
            {
                if (fabs(s[i]) > temp)
                {
                    temp = fabs(s[i]);
                    ik = i;
                }
            }

            // exchange rows
            if (ik != k)
            {
                temp = s[k];
                s[k] = s[ik];
                s[ik] = temp;

                for (int t = 0; t <= k - 1; ++t)
                {
                    temp = L[k][t];
                    L[k][t] = L[ik][t];
                    L[ik][t] = temp;
                }

                for (int t = k; t < N; ++t)
                {
                    temp = A[k][t];
                    A[k][t] = A[ik][t];
                    A[ik][t] = temp;
                }

                for (int t = 0; t < N; ++t)
                {
                    temp = Q[k][t];
                    Q[k][t] = Q[ik][t];
                    Q[ik][t] = temp;
                }
            }

            U[k][k] = s[k];
            for (int j = k + 1; j < N; ++j)
            {
                U[k][j] = A[k][j];
                for (int t = 0; t <= k - 1; ++t)
                {
                    U[k][j] -= L[k][t] * U[t][j];
                }
            }

            L[k][k] = 1.0;
            for (int i = k + 1; i < N; ++i)
            {
                L[i][k] = s[i] / U[k][k];
            }
        }
    }
};