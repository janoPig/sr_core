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

    struct BatchScore
    {
        size_t mIndex;
        double mScore;
    };

    template <size_t BATCH>
    struct Result
    {
        Result() = default;

        inline double Mean() const noexcept
        {
            return mScoreSum / (mScore.size() * BATCH);
        }

        inline void Reset() noexcept
        {
            mScoreSum = 0.0;
            mScore.clear();
        }

        inline void Add(size_t batchIndex, double score) noexcept
        {
            mScoreSum += score;
            mScore.push_back({batchIndex, score});
        }

        void GetNWorst(size_t n, std::vector<BatchScore> &sel)
        {
            n = std::min(mScore.size(), n);
            sel.resize(n);
            std::partial_sort_copy(mScore.begin(), mScore.end(), sel.begin(), sel.end(), [](const auto &a, const auto &b)
                                   { return a.mScore > b.mScore; });
        }

        double mScoreSum{0.0};
        std::vector<BatchScore> mScore;
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

    template <typename T, bool SW>
    double ComputeSqErr(const T *const __restrict yTrue,
                        const T *const __restrict yPred,
                        const size_t size,
                        const T *const __restrict sampleWeight = nullptr) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            auto _err = (yPred[n] - yTrue[n]) * (yPred[n] - yTrue[n]);
            if constexpr (SW)
            {
                _err *= sampleWeight[n];
            }
            err += _err;
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T, bool SW>
    double ComputeMAE(const T *const __restrict yTrue,
                      const T *const __restrict yPred,
                      const size_t size,
                      const T *const __restrict sampleWeight = nullptr) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            auto _err = std::abs(yPred[n] - yTrue[n]);
            if constexpr (SW)
            {
                _err *= sampleWeight[n];
            }
            err += std::abs(yPred[n] - yTrue[n]);
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T, bool SW>
    double ComputeMSLE(const T *const __restrict yTrue,
                       const T *const __restrict yPred,
                       const size_t size,
                       const T *const __restrict sampleWeight = nullptr) noexcept
    {
        double err{};
        for (size_t n = 0; n < size; n++)
        {
            auto _err = std::log(1.0 + yTrue[n]) - std::log(1.0 + yPred[n]);
            _err *= _err;
            if constexpr (SW)
            {
                _err *= sampleWeight[n];
            }
            err += _err;
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T, bool CW, bool SW>
    double ComputeLogLoss(const T *const __restrict yTrue,
                          const T *const __restrict yPred,
                          const size_t size,
                          T cw0 = (T)1.0,
                          T cw1 = (T)1.0,
                          const T *const __restrict sampleWeight = nullptr) noexcept
    {
        T err{};
        for (size_t n = 0; n < size; n++)
        {
            if (yTrue[n] > (T)0.999999)
            {
                auto _err = -std::log(yPred[n]);
                if constexpr (CW)
                {
                    _err *= cw1;
                }
                if constexpr (SW)
                {
                    _err *= sampleWeight[n];
                }
                err += _err;
            }
            else
            {
                auto _err = -std::log(static_cast<T>(1.0) - yPred[n]);
                if constexpr (CW)
                {
                    _err *= cw0;
                }
                if constexpr (SW)
                {
                    _err *= sampleWeight[n];
                }
                err += _err;
            }
        }

        if (IsFinite(err))
        {
            return static_cast<double>(err);
        }
        return LARGE_FLOAT;
    }

    template <typename T, bool CW, bool SW>
    double ComputeLogitApprox(const T *const __restrict yTrue,
                              const T *const __restrict yPred,
                              const size_t size,
                              T cw0 = (T)1.0,
                              T cw1 = (T)1.0,
                              const T *const __restrict sampleWeight = nullptr) noexcept
    {
        T err{};
        const T cw[] = {cw0, cw1};
        for (size_t n = 0; n < size; n++)
        {
            auto y = std::max(yPred[n], static_cast<T>(-5.0));
            y = std::min(y, static_cast<T>(5.0));
            const auto t0 = y * (static_cast<T>(0.5) - yTrue[n]) + static_cast<T>(0.69314718056);
            const auto t1 = y * y;
            auto _err = t0 + static_cast<T>(5.36669250834) * t1 / (t1 + static_cast<T>(49.1441046693));
            if constexpr (CW)
            {
                _err *= cw[yTrue[n] > static_cast<T>(0.5)];
            }
            if constexpr (SW)
            {
                _err *= sampleWeight[n];
            }
            err += _err;
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
                y[i] = std::max(y[i], static_cast<T>(-20.0));
                y[i] = std::min(y[i], static_cast<T>(20.0));
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