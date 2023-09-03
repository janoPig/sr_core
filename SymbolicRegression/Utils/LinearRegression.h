#pragma once

namespace SymbolicRegression::Utils
{
    // https://github.com/willnode/N-Matrix-Programmer

    template <typename T>
    bool matrix_inverse_2(const T *__restrict M, T *__restrict I) noexcept
    {
        auto det = M[0] * M[3] - M[1] * M[2];
        if (det == static_cast<T>(0.0))
            return false;
        det = static_cast<T>(1.0) / det;

        I[0] = det * M[3];
        I[1] = -det * M[1];
        I[2] = -det * M[2];
        I[3] = det * M[0];

        return true;
    }

    template <typename T>
    bool matrix_inverse_3(const T *__restrict M, T *__restrict I) noexcept
    {
        /*
        00 01 02
        10 11 12
        20 21 22
        */
        auto det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
        if (det == static_cast<T>(0.0))
            return false;
        det = 1 / det;

        I[0] = det * (M[4] * M[8] - M[5] * M[7]);
        I[1] = det * (M[2] * M[7] - M[1] * M[8]);
        I[2] = det * (M[1] * M[5] - M[2] * M[4]);
        I[3] = det * (M[5] * M[6] - M[3] * M[8]);
        I[4] = det * (M[0] * M[8] - M[2] * M[6]);
        I[5] = det * (M[2] * M[3] - M[0] * M[5]);
        I[6] = det * (M[3] * M[7] - M[4] * M[6]);
        I[7] = det * (M[1] * M[6] - M[0] * M[7]);
        I[8] = det * (M[0] * M[4] - M[1] * M[3]);

        return true;
    }

    template <typename T, int DIMENSION>
    bool matrix_inverse(const T *M, T *I) noexcept
    {
        static_assert(DIMENSION > 1 && DIMENSION < 4);
        if constexpr (DIMENSION == 2)
            return matrix_inverse_2(M, I);
        if constexpr (DIMENSION == 3)
            return matrix_inverse_3(M, I);

        return false;
    }

    template <typename T>
    void normal_matrix_2(const T *__restrict X[2], T *__restrict N, size_t size) noexcept
    {
        for (auto i = 0; i < 4; i++)
            N[i] = static_cast<T>(0.0);

        if (X[0])
        {
            for (size_t i = 0; i < size; i++)
            {
                N[0] += X[0][i] * X[0][i];
                N[1] += X[0][i] * X[1][i];
                N[2] += X[1][i] * X[0][i];
                N[3] += X[1][i] * X[1][i];
            }
        }
        else
        {
            N[0] = static_cast<T>(size);
            for (size_t i = 0; i < size; i++)
            {
                N[1] += X[1][i];
                N[3] += X[1][i] * X[1][i];
            }
            N[2] = N[1];
        }
    }

    template <typename T>
    void moment_matrix_2(const T *__restrict X[2], const T *__restrict y, T *__restrict M, size_t size) noexcept
    {
        for (auto i = 0; i < 2; i++)
            M[i] = static_cast<T>(0.0);

        if (X[0])
        {
            for (size_t i = 0; i < size; i++)
            {
                M[0] += X[0][i] * y[i];
                M[1] += X[1][i] * y[i];
            }
        }
        else
        {
            for (size_t i = 0; i < size; i++)
            {
                M[0] += y[i];
                M[1] += X[1][i] * y[i];
            }
        }
    }

    template <typename T>
    bool coefficients_2(const T *__restrict N, const T *__restrict M, T *B) noexcept
    {
        T invN[4];
        if (!matrix_inverse_2(N, invN))
            return false;

        B[0] = invN[0] * M[0] + invN[1] * M[1];
        B[1] = invN[2] * M[0] + invN[3] * M[1];

        return true;
    }

    template <typename T>
    bool linear_regression_2(const T *__restrict X[2], const T *__restrict y, T *B, size_t size) noexcept
    {
        T N[4];
        normal_matrix_2<T>(X, N, size);
        T M[2];
        moment_matrix_2<T>(X, y, M, size);
        return coefficients_2(N, M, B);
    }
}