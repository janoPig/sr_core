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

    template <typename T>
    double ComputeLogLoss(const T *const __restrict yTrue, const T *const __restrict yPred, const size_t size) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            err += -(yTrue[n] * std::log(yPred[n]) + (static_cast<T>(1.0) - yTrue[n]) * std::log(static_cast<T>(1.0) - yPred[n]));
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