#pragma once

namespace SymbolicRegression::Utils
{
    // total sum of square
    template <typename T>
    double TSS(const T *data, size_t size) noexcept
    {
        double avg_y{0.0};
        for (size_t i = 0; i < size; i++)
        {
            avg_y += static_cast<double>(data[i]);
        }
        avg_y /= size;

        double tss{0.0};
        for (size_t i = 0; i < size; i++)
        {
            const double val = static_cast<double>(data[i]);
            tss += (val - avg_y) * (val - avg_y);
        }
        return tss / size;
    }

    constexpr auto R2(double mse, double tss) noexcept
    {
        return 1.0 - mse / tss;
    }

    struct Result
    {
        Result() = default;

        inline double mean() const noexcept
        {
            return mScore / mSamplesCount;
        }

        inline void reset() noexcept
        {
            mSamplesCount = 0;
            mScore = 0.0;
        }

        size_t mSamplesCount{0};
        double mScore{0.0};
    };

    template <typename T, size_t S>
    void Clip(T *y, const T minVal, const T maxVal) noexcept
    {
        for (size_t n = 0; n < S; n++)
        {
            y[n] = std::max(y[n], minVal);
            y[n] = std::min(y[n], maxVal);
        }
    }

    template <typename T>
    double ComputeSqErr(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += (yPred[n] - yTrue[n]) * (yPred[n] - yTrue[n]);
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    double ComputeMAE(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += std::abs(yPred[n] - yTrue[n]);
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    double ComputeMSLE(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        double err{};
        for (size_t n = 0; n < size; n++)
        {
            const auto x = std::log(1.0 + yTrue[n]) - std::log(1.0 + yPred[n]);
            err += x * x;
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T, bool CW>
    double ComputeLogLoss(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size, T cw0 = (T)1.0, T cw1 = (T)1.0) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            if (yTrue[n] > (T)0.999999)
            {
                if constexpr (CW)
                {
                    err += cw1 * (-std::log(yPred[n]));
                }
                else
                {
                    err += -std::log(yPred[n]);
                }
                // err += -(yTrue[n] * std::log(yPred[n]) + (static_cast<T>(1.0) - yTrue[n]) * std::log(static_cast<T>(1.0) - yPred[n]));
            }
            else
            {
                if constexpr (CW)
                {
                    err += -cw0 * std::log(static_cast<T>(1.0) - yPred[n]);
                }
                else
                {
                    err += -std::log(static_cast<T>(1.0) - yPred[n]);
                }
            }
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    constexpr T __log_test_1(const T x) noexcept
    {
        const auto tmp_0 = (T)0.00785912107676267624 + x;
        const auto tmp_1 = x / tmp_0;
        const auto tmp_2 = tmp_1 + tmp_1;
        const auto tmp_3 = (T)0.02958619594573974609 + tmp_2;
        const auto tmp_4 = x * (T)18.00337600708007812500;
        const auto tmp_6 = ((T)-4.69269418716430664062) / tmp_3;
        const auto tmp_7 = x + (T)0.00015384485595859587;
        const auto tmp_8 = tmp_6 + ((T)-118.58383941650390625000);
        const auto tmp_9 = tmp_8 * (T)0.05831270664930343628;
        const auto tmp_10 = tmp_3 + tmp_9;
        const auto tmp_11 = tmp_3 + tmp_10;
        const auto tmp_12 = tmp_1 + tmp_7;
        const auto tmp_13 = tmp_3 + x;
        const auto tmp_14 = tmp_12 * tmp_13;
        const auto tmp_15 = tmp_4 / tmp_14;
        const auto tmp_16 = tmp_11 + tmp_15;
        return tmp_16;
    }

    template <typename T>
    constexpr T __log_test_2(const T x) noexcept
    {
        const auto tmp_0 = x + x;
        const auto tmp_1 = tmp_0 + (T)0.15853951871395111084;
        const auto tmp_2 = x / tmp_1;
        const auto tmp_3 = tmp_2 + (T)0.00299614411778748035;
        const auto tmp_4 = tmp_2 + x;
        const auto tmp_5 = tmp_4 + (T)0.19225053489208221436;
        const auto tmp_6 = x / tmp_3;
        const auto tmp_8 = tmp_5 + tmp_6;
        const auto tmp_9 = (T)0.00000174532362962054 + x;
        const auto tmp_10 = x / tmp_9;
        const auto tmp_11 = tmp_1 * x;
        const auto tmp_12 = tmp_10 + tmp_11;
        const auto tmp_13 = tmp_12 - (T)3.06196260452270507812;
        const auto tmp_14 = tmp_13 / tmp_8;
        return tmp_14;
    }

    template <typename T>
    constexpr T __log_test_3(const T x) noexcept
    {
        const auto tmp_0 = (T)0.00000228071621677373 + x;
        const auto tmp_1 = x / tmp_0;
        // tmp_2 = 0.10014304518699645996 + (-3.08318519592285156250);
        const auto tmp_3 = tmp_0 / (T)0.18211431801319122314;
        const auto tmp_4 = tmp_3 + (T)0.00424051098525524139;
        const auto tmp_5 = tmp_1 - (T)2.98304215073585510254; //+ tmp_2;
        const auto tmp_6 = (T)0.18998736143112182617 + tmp_3;
        const auto tmp_7 = x / tmp_4;
        const auto tmp_8 = tmp_7 + tmp_6;
        const auto tmp_9 = tmp_5 / tmp_8;
        return tmp_9;
    }

    template <typename T>
    double ApproxLogLoss_1(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += -(yTrue[n] * __log_test_1(yPred[n]) + (static_cast<T>(1.0) - yTrue[n]) * __log_test_1(static_cast<T>(1.0) - yPred[n]));
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    double ApproxLogLoss_2(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += -(yTrue[n] * __log_test_2(yPred[n]) + (static_cast<T>(1.0) - yTrue[n]) * __log_test_2(static_cast<T>(1.0) - yPred[n]));
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    double ApproxLogLoss_3(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += -(yTrue[n] * __log_test_3(yPred[n]) + (static_cast<T>(1.0) - yTrue[n]) * __log_test_3(static_cast<T>(1.0) - yPred[n]));
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    constexpr T __logit_test(const T x) noexcept
    {
        if (x == 0)
            return 0.69314718056f;
        else if (x < -20)
            return -x;
        else if (x > 20)
            return 0.0;
        const auto tmp_0 = x * x;
        const auto tmp_1 = 22.52812767028808593750 + tmp_0;
        const auto tmp_2 = x * 0.00897947046905755997;
        const auto tmp_3 = tmp_0 * tmp_2;
        const auto tmp_5 = tmp_3 + x;
        const auto tmp_6 = tmp_5 + x;
        const auto tmp_7 = tmp_6 / tmp_1;
        const auto tmp_9 = tmp_7 / tmp_2;
        const auto tmp_10 = tmp_9 + tmp_9;
        const auto tmp_11 = (-53.71286010742187500000) / tmp_10;
        const auto tmp_12 = (-0.25884103775024414062) + tmp_7;
        const auto tmp_15 = tmp_11 + x;
        const auto tmp_17 = tmp_15;
        const auto tmp_19 = tmp_17 * tmp_12;
        return tmp_19;
    }

    template <typename T, bool CW>
    __attribute__((noinline)) double ApproxLogit(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size, T cw0 = (T)1.0, T cw1 = (T)1.0) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            if (yTrue[n] > (T)0.999999)
            {
                if constexpr (CW)
                {
                    err += cw1 * __logit_test(yPred[n]);
                }
                else
                {
                    err += __logit_test(yPred[n]);
                }
            }
            else
            {
                if constexpr (CW)
                {
                    err += cw0 * __logit_test(-yPred[n]);
                }
                else
                {
                    err += __logit_test(-yPred[n]);
                }
            }
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T>
    double ComputePseudoKendall(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t j = 1; j < size; j++)
        {
            for (size_t i = 0; i < j; i++)
            {
                const auto t = (yTrue[i] - yTrue[j]) * (yPred[i] - yPred[j]);
                err += t ? t / std::abs(t) : yTrue[i] - yTrue[j] == 0 ? static_cast<T>(1.0)
                                                                      : static_cast<T>(0.0);
            }
        }
        err *= static_cast<T>(2.0) / (size * (size - 1));
        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return 0.0;
    }

    template <typename T, size_t S>
    void TransformData(T *y, uint32_t transformation) noexcept
    {
        if (transformation == 1)
        {
            for (size_t i = 0; i < S; i++)
            {
                y[i] = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-y[i]));
            }
        }
        else if (transformation == 2)
        {
            for (size_t i = 0; i < S; i++)
            {
                y[i] = static_cast<T>(0.25) * y[i] + static_cast<T>(0.5);
                y[i] = std::max(y[i], static_cast<T>(0.0));
                y[i] = std::min(y[i], static_cast<T>(1.0));
            }
        }
        else if (transformation == 3)
        {
            for (size_t i = 0; i < S; i++)
            {
                y[i] = std::round(y[i]);
            }
        }
    }
}