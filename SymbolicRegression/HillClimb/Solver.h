#pragma once

#include "Mutation.h"
#include "HillClimber.h"
#include "CodeInitializer.h"
#include "../Utils/Dataset.h"
#include "../Computer/Machine.h"

using namespace std::chrono;
namespace SymbolicRegression::HillClimb
{
    struct CodeInfo
    {
        double mScore;
        double mPartialScore;
        std::string mEquation;
        std::string mCode;
        std::vector<double> mConstants;
    };

    template <typename T, size_t BATCH, bool DBG = false>
    class Solver
    {
        using Code = Computer::Code<T>;
        using EvCode = EvaluatedCode<T>;
        using Storage = std::vector<EvCode>;
        using Dataset = Utils::Dataset<T, BATCH>;
        using INSTR_SET = Computer::Instructions::Set;
        using Machine = Computer::Machine<T, BATCH>;

    public:
        Solver() = delete;
        ~Solver() = default;

        Solver(const Config &config)
            : mInitialized(false),
              mConfig(config),
              mRandom(),
              mMachine(config.mCodeSettings),
              mPopulation(config.mPopulationSize, config.mCodeSettings),
              mBestCode(config.mCodeSettings)
        {
            mRandom.Seed(config.mRandomSeed);
        }

        template <typename CALLBACK>
        double Fit(const Dataset &data,
                   const FitParams &fp,
                   CALLBACK &&callback,
                   const Utils::BatchVector<T, BATCH> *sampleWeight)
        {
            if (fp.mVerbose > 1)
                printf("fit start, random engine state %zu\n", mRandom.State());

            const auto fitStartTime = high_resolution_clock::now();

            if (!mInitialized)
            {
                Initialize(data, fp, sampleWeight);
            }

            if (fp.mVerbose > 1)
                callback(0, mBestCode.mScore[2]);

            const CodeMutation codeMut{fp.mConstSettings, fp.mInstrProbs, fp.mFeatProbs, mRandom};
            const ConstMutation<T> constMut{mRandom, fp.mConstSettings};

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            EvCode neighbour(mConfig.mCodeSettings);
            std::vector<size_t> sel0(1);

            size_t it{};
            while (true)
            {
                it++;
                if (fp.mIterLimit && it > fp.mIterLimit)
                {
                    if (fp.mVerbose > 1)
                        printf("iter limit reached! it: %zu\n", it - 1);
                    break;
                }
                if (fp.mTimeLimit && it % 100 == 0)
                {
                    const auto duration = (uint64_t)duration_cast<milliseconds>(high_resolution_clock::now() - fitStartTime).count();
                    if (duration >= fp.mTimeLimit)
                    {
                        if (fp.mVerbose > 1)
                            printf("duration or criteria limit reached! it: %zu dur: %zu\n", it, duration);
                        break;
                    }
                }
                if (it % 10000 == 0)
                {
                    const auto score = EvalPopulation(data, fp, sampleWeight);
                    std::cout << mBestCode.mCode.mTreeComplexity << " " << GetExpression(mBestCode) << std::endl;
                    if (fp.mVerbose > 1)
                        callback(it, score);
                }

                const auto [hillclimber, selIdx] = TournamentSelection(fp.mTournament);

                auto bestCode = hillclimber->Current();
                auto bestScore = LARGE_FLOAT;
                bool find = false;
                std::pair<size_t, double> worstBatch;
                sel0[0] = hillclimber->mWorstBatch.first;

                for (int subStep = 0; subStep < 15; subStep++)
                {
                    neighbour = hillclimber->Current();
                    neighbour.ResetScore();

                    for (int muteStep = 0; muteStep < 1; muteStep++)
                    {
                        codeMut(neighbour.mCode);
                        constMut(neighbour.mCode);

                        if (neighbour.mCode.IsConstExpression(indices.data(), mCodeMapping.set))
                            continue;

                        Evaluate(data, neighbour, sel0, 0, fp, sampleWeight);

                        if (neighbour.mScore[0] >= 1.15 * hillclimber->Current().mScore[0])
                        {
                            continue;
                        }

                        auto tmp = Evaluate(data, neighbour, hillclimber->mSample, 1, fp, sampleWeight);

                        if (neighbour.mScore[1] < bestScore)
                        {
                            bestCode = neighbour;
                            bestScore = neighbour.mScore[1];
                            worstBatch = tmp;
                            find = true;
                        }
                    }
                }

                if (find)
                {
                    if (bestCode.mScore[1] < hillclimber->Best().mScore[1] * 1.15)
                    {
                        hillclimber->Current() = bestCode;
                        if (bestCode.mScore[1] < hillclimber->Best().mScore[1])
                        {
                            bestCode.mScore[0] = worstBatch.second;
                            hillclimber->Best() = bestCode;
                            hillclimber->mWorstBatch = worstBatch;
                        }
                    }
                }
            }
            return EvalPopulation(data, fp, sampleWeight);
        }

        void Predict(Dataset &data, uint32_t transformation, T clipMin, T clipMax) noexcept
        {
            mMachine.Compute(data, mBestCode.mCode, transformation, clipMin, clipMax);
        }

        void Predict(Dataset &data, uint32_t transformation, uint32_t id, T clipMin, T clipMax) noexcept
        {
            auto &hc = mPopulation[id];
            mMachine.Compute(data, hc.Best().mCode, transformation, clipMin, clipMax);
        }

        const auto &GetBestCode()
        {
            return mBestCode;
        }

        double EvalPopulation(const Dataset &data,
                              const FitParams &fp,
                              const Utils::BatchVector<T, BATCH> *sampleWeight,
                              double alpha = 0.05) noexcept
        {
            auto bestScore = mBestCode.mScore[2];
            for (auto &hc : mPopulation)
            {
                if (hc.Best().mScore[1] > (1.0 + alpha) * bestScore)
                    continue;
                hc.Best().mScore[2] = Evaluate(data, hc.Best(), fp, sampleWeight);
                if (hc.Best().mScore[2] < bestScore)
                {
                    bestScore = hc.Best().mScore[2];
                    mBestCode = hc.Best();
                }
            }
            return mBestCode.mScore[2];
        }

        double Score() const noexcept
        {
            return mBestCode.mScore[2];
        }

        CodeInfo GetBestInfo() noexcept
        {
            return CodeInfo{mBestCode.mScore[2], mBestCode.mScore[1], GetExpression(mBestCode), GenerateCode(mBestCode, "equation"), mBestCode.mCode.GetConstants()};
        }

        CodeInfo GetInfo(size_t threadIdx, size_t idx) noexcept
        {
            auto &code = mPopulation[idx].Best();
            const std::string eq_name = "equation_" + std::to_string(threadIdx) + "_" + std::to_string(idx);
            return CodeInfo{code.mScore[2], code.mScore[1], GetExpression(code), GenerateCode(code, eq_name), code.mCode.GetConstants()};
        }

        std::string GetExpression(EvCode &c) noexcept
        {
            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            c.mCode.IsConstExpression(indices.data(), mCodeMapping.set);
            auto str = c.mCode.GetString(mCodeMapping.set);

            return str;
        }

        std::string GenerateCode(EvCode &c, const std::string &eqName) noexcept
        {
            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            c.mCode.IsConstExpression(indices.data(), mCodeMapping.set);
            auto str = c.mCode.GenerateCode(mCodeMapping.set, eqName);

            return str;
        }

        const Config &GetConfig() const noexcept
        {
            return mConfig;
        }

    private:
        void Initialize(const Dataset &data, const FitParams &fp, const Utils::BatchVector<T, BATCH> *sampleWeight)
        {
            assert(!mInitialized);

            const CodeInitializer<T> codeInit{mConfig.mCodeSettings, mConfig.mInitConstSettings, fp.mInstrProbs, fp.mFeatProbs, mRandom};
            auto sampleSize = std::min((size_t)16, data.BatchCount());

            mFullSet.resize(data.BatchCount());
            std::iota(mFullSet.begin(), mFullSet.end(), 0);
            auto allSamples = mFullSet;

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            std::vector<size_t> bs(1);
            bs[0] = mRandom.Rand(data.BatchCount());

            for (auto &hc : mPopulation)
            {
                hc.mSample.resize(sampleSize);
                if (sampleSize == data.BatchCount())
                {
                    std::iota(hc.mSample.begin(), hc.mSample.end(), 0);
                }
                else
                {
                    mRandom.Shuffle(allSamples.begin(), allSamples.end());
                    for (size_t i = 0; i < hc.mSample.size(); i++)
                    {
                        hc.mSample[i] = allSamples[i];
                    }
                }

                EvCode candidate{mConfig.mCodeSettings};

                int cnt = 3;
                int k = 30;
                auto bestScore = LARGE_FLOAT;
                while (cnt && k)
                {
                    codeInit.operator()(candidate.mCode);

                    if (!candidate.mCode.IsConstExpression(indices.data(), mCodeMapping.set))
                    {
                        Evaluate(data, candidate, bs, 0, fp, sampleWeight);

                        if (cnt == 3 || candidate.mScore[0] < bestScore)
                        {
                            bestScore = candidate.mScore[0];
                            hc.Current() = candidate;
                            cnt--;
                        }
                    }
                    k--;
                }
                auto &current = hc.Current();
                hc.mWorstBatch = Evaluate(data, current, hc.mSample, 1, fp, sampleWeight);
                current.mScore[0] = hc.mWorstBatch.second;
                hc.Best() = hc.Current();

                if (hc.Best().mScore[1] < mBestCode.mScore[1])
                {
                    mBestCode = hc.Current();
                }
            }

            mInitialized = true;
        }

        auto Evaluate(const Dataset &data,
                      EvCode &evc,
                      const std::vector<size_t> &batchSelection,
                      int id,
                      const FitParams &fp,
                      const Utils::BatchVector<T, BATCH> *sampleWeight) noexcept
        {
            Utils::Result r;
            auto x = mMachine.ComputeScore(data, evc.mCode, batchSelection, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, (T)fp.mClassWeights[0], (T)fp.mClassWeights[1], true, sampleWeight);
            evc.mScore[id] = r.mean();
            return x;
        }

        auto Evaluate(const Dataset &data,
                      const EvCode &evc,
                      const FitParams &fp,
                      const Utils::BatchVector<T, BATCH> *sampleWeight) noexcept
        {
            Utils::Result r;
            [[maybe_unused]] const auto x = mMachine.ComputeScore(data, evc.mCode, mFullSet, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, (T)fp.mClassWeights[0], (T)fp.mClassWeights[1], true, sampleWeight);
            return r.mean();
        }

        auto TournamentSelection(size_t tournament = 1) noexcept
        {
            size_t bestIdx = 0;
            double bestFit = std::numeric_limits<double>::max();
            for (size_t i = 0; i < tournament; i++)
            {
                const auto idx = mRandom.Rand(mPopulation.size());
                const auto &tmp = mPopulation[idx];
                if (tmp.Best().mScore[1] < bestFit)
                {
                    bestFit = tmp.Best().mScore[1];
                    bestIdx = idx;
                }
            }
            return std::pair{&mPopulation[bestIdx], bestIdx};
        }

    private:
        bool mInitialized;
        Config mConfig;
        Utils::RandomEngine mRandom;

        const Computer::CodeGen::CodeMapping<INSTR_SET> mCodeMapping;

        Machine mMachine;
        std::vector<HillClimber<T>> mPopulation;
        EvCode mBestCode;

        std::vector<size_t> mFullSet;
    };
}
