#pragma once

#include "../SymbolicRegression/SymbolicRegression.h"

namespace Hroch
{
    using namespace SymbolicRegression;
    constexpr static size_t BATCH = 64;

    enum class EDataType
    {
        F32,
        F64
    };

    using DataSetF = Utils::Dataset<float, BATCH>;
    using DataSetD = Utils::Dataset<double, BATCH>;

    using SolverF = HillClimb::Solver<float, BATCH, false>;
    using SolverD = HillClimb::Solver<double, BATCH, false>;

    using SampleWeightF = Utils::BatchVector<float, BATCH>;
    using SampleWeightD = Utils::BatchVector<double, BATCH>;

    class ISolver
    {
    public:
        virtual ~ISolver() = default;
        virtual void Fit(const DataSetF &, const FitParams &, const SampleWeightF *) = 0;
        virtual void Fit(const DataSetD &, const FitParams &, const SampleWeightD *) = 0;
        virtual void Predict(DataSetF &, uint32_t, float, float) = 0;
        virtual void Predict(DataSetD &, uint32_t, double, double) = 0;
        virtual void Predict(DataSetF &, uint32_t, uint32_t, float, float) = 0;
        virtual void Predict(DataSetD &, uint32_t, uint32_t, double, double) = 0;
        virtual double Score() const noexcept = 0;
        virtual HillClimb::CodeInfo GetBestInfo() = 0;
        virtual HillClimb::CodeInfo GetInfo(size_t threadId, size_t idx) noexcept = 0;
        virtual const Config &GetConfig() const noexcept = 0;
    };

    template <typename SolverType, EDataType DataType>
    class SolverWrapper : public ISolver, private SolverType
    {
        friend class SolverFactory;

    protected:
        SolverWrapper(const Config &cfg)
            : SolverType(cfg)
        {
        }

        static void test_callback([[maybe_unused]] const uint64_t it, [[maybe_unused]] const double err) noexcept
        {
            // printf("[%zu] %f", it, err);
        }

    public:
        virtual ~SolverWrapper() = default;

        void Fit(const DataSetF &data, const FitParams &fp, const SampleWeightF *sw) override
        {

            if constexpr (DataType == EDataType::F32)
            {
                SolverType::Fit(data, fp, test_callback, sw);
            }
        }

        void Fit(const DataSetD &data, const FitParams &fp, const SampleWeightD *sw) override
        {
            if constexpr (DataType == EDataType::F64)
            {
                SolverType::Fit(data, fp, test_callback, sw);
            }
        }

        void Predict(DataSetF &data, uint32_t transformation, float clipMin, float clipMax) noexcept override
        {
            if constexpr (DataType == EDataType::F32)
            {
                SolverType::Predict(data, transformation, clipMin, clipMax);
            }
        }

        void Predict(DataSetD &data, uint32_t transformation, double clipMin, double clipMax) noexcept override
        {
            if constexpr (DataType == EDataType::F64)
            {
                SolverType::Predict(data, transformation, clipMin, clipMax);
            }
        }

        void Predict(DataSetF &data, uint32_t transformation, uint32_t id, float clipMin, float clipMax) noexcept override
        {
            if constexpr (DataType == EDataType::F32)
            {
                SolverType::Predict(data, transformation, id, clipMin, clipMax);
            }
        }

        void Predict(DataSetD &data, uint32_t transformation, uint32_t id, double clipMin, double clipMax) noexcept override
        {
            if constexpr (DataType == EDataType::F64)
            {
                SolverType::Predict(data, transformation, id, clipMin, clipMax);
            }
        }

        double Score() const noexcept override
        {
            return SolverType::Score();
        }

        HillClimb::CodeInfo GetBestInfo() noexcept override
        {
            return SolverType::GetBestInfo();
        }

        HillClimb::CodeInfo GetInfo(size_t threadId, size_t idx) noexcept override
        {
            return SolverType::GetInfo(threadId, idx);
        }

        const Config &GetConfig() const noexcept override
        {
            return SolverType::GetConfig();
        }
    };
    class SolverFactory
    {
    public:
        static ISolver *Create(const Config &cfg, EDataType dataType)
        {
            ISolver *solver = nullptr;
            if (dataType == EDataType::F32)
                solver = new SolverWrapper<SolverF, EDataType::F32>(cfg);
            if (dataType == EDataType::F64)
                solver = new SolverWrapper<SolverD, EDataType::F64>(cfg);
            return solver;
        }
    };
}
