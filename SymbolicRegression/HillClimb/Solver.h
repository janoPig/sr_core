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

            const CodeMutation codeMut{fp.mBeta, fp.mConstSettings, fp.mInstrProbs, fp.mFeatProbs, mRandom};
            const ConstMutation<T> constMut{mRandom, fp.mConstSettings};

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            EvCode neighbour(mConfig.mCodeSettings);
            std::vector<size_t> sel0(fp.mPretestSize);

            Utils::Result<BATCH> r;
            std::vector<Utils::BatchScore> worstBatches;

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
                    if (fp.mVerbose > 1)
                    {
                        const auto score = EvalPopulation(data, fp, sampleWeight);
                        callback(it, score);
                        std::cout << mBestCode.mCode.mTreeComplexity << " " << GetExpression(mBestCode) << std::endl;
                    }
                }

                const auto [hillclimber, selIdx] = TournamentSelection(fp.mTournament);

                auto bestCode = hillclimber->Current();
                auto bestScore = LARGE_FLOAT;
                bool find = false;

                for (size_t i = 0; i < hillclimber->mPretest.size(); i++)
                {
                    sel0[i] = hillclimber->mPretest[i].mIndex;
                }
                Utils::Result<BATCH> r;

                for (uint32_t subStep = 0; subStep < fp.mNeighboursCount; subStep++)
                {
                    neighbour = hillclimber->Current();
                    neighbour.ResetScore();

                    for (int muteStep = 0; muteStep < 1; muteStep++)
                    {
                        codeMut(neighbour.mCode);
                        constMut(neighbour.mCode);

                        if (neighbour.mCode.IsConstExpression(indices.data(), mCodeMapping.set))
                            continue;

                        Evaluate(data, neighbour, sel0, 0, fp, sampleWeight, r);

                        if (neighbour.mScore[0] >= (1.0 + fp.mAlpha) * hillclimber->Current().mScore[0])
                        {
                            continue;
                        }

                        Evaluate(data, neighbour, hillclimber->mSample, 1, fp, sampleWeight, r);

                        if (neighbour.mScore[1] < bestScore)
                        {
                            bestCode = neighbour;
                            bestScore = neighbour.mScore[1];
                            r.GetNWorst(fp.mPretestSize, worstBatches);
                            find = true;
                        }
                    }
                }

                if (find)
                {
                    if (bestCode.mScore[1] < hillclimber->Best().mScore[1] * (1.0 + fp.mAlpha))
                    {
                        hillclimber->Current() = bestCode;
                        if (bestCode.mScore[1] < hillclimber->Best().mScore[1])
                        {
                            bestCode.mScore[0] = GetScore(worstBatches);
                            hillclimber->Best() = bestCode;
                            hillclimber->mPretest = worstBatches;
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
            Utils::Result<BATCH> r;
            for (auto &hc : mPopulation)
            {
                if (hc.Best().mScore[1] > (1.0 + alpha) * bestScore)
                    continue;
                hc.Best().mScore[2] = EvaluateAll(data, hc.Best(), fp, sampleWeight, r);
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
            const auto sampleSize = std::min((size_t)fp.mSampleSize, data.BatchCount());
            const auto pretestSize = std::min((size_t)fp.mPretestSize, data.BatchCount());

            mFullSet.resize(data.BatchCount());
            std::iota(mFullSet.begin(), mFullSet.end(), 0);
            auto allSamples = mFullSet;

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            auto selectSample = [&allSamples, &data, this](auto size, auto &s)
            {
                s.resize(size);
                if (size == data.BatchCount())
                {
                    std::iota(s.begin(), s.end(), 0);
                }
                else
                {
                    mRandom.Shuffle(allSamples.begin(), allSamples.end());
                    for (size_t i = 0; i < size; i++)
                    {
                        s[i] = allSamples[i];
                    }
                }
            };

            std::vector<size_t>
                pretest(pretestSize);
            selectSample(pretestSize, pretest);
            Utils::Result<BATCH> r;

            for (auto &hc : mPopulation)
            {
                selectSample(sampleSize, hc.mSample);
                EvCode candidate{mConfig.mCodeSettings};

                int cnt = 3;
                int k = 30;
                auto bestScore = LARGE_FLOAT;
                while (cnt && k)
                {
                    codeInit.operator()(candidate.mCode);

                    if (!candidate.mCode.IsConstExpression(indices.data(), mCodeMapping.set))
                    {
                        Evaluate(data, candidate, pretest, 0, fp, sampleWeight, r);

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
                Evaluate(data, current, hc.mSample, 1, fp, sampleWeight, r);
                r.GetNWorst(pretestSize, hc.mPretest);
                current.mScore[0] = GetScore(hc.mPretest);
                hc.Best() = hc.Current();

                if (hc.Best().mScore[1] < mBestCode.mScore[1])
                {
                    mBestCode = hc.Current();
                }
            }

            mInitialized = true;
        }

        void Evaluate(const Dataset &data,
                      EvCode &evc,
                      const std::vector<size_t> &batchSelection,
                      int id,
                      const FitParams &fp,
                      const Utils::BatchVector<T, BATCH> *sampleWeight,
                      Utils::Result<BATCH> &r) noexcept
        {
            r.Reset();
            mMachine.ComputeScore(data, evc.mCode, batchSelection, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, (T)fp.mClassWeights[0], (T)fp.mClassWeights[1], true, sampleWeight);
            evc.mScore[id] = r.Mean();
        }

        auto EvaluateAll(const Dataset &data,
                         const EvCode &evc,
                         const FitParams &fp,
                         const Utils::BatchVector<T, BATCH> *sampleWeight,
                         Utils::Result<BATCH> &r) noexcept
        {
            r.Reset();
            mMachine.ComputeScore(data, evc.mCode, mFullSet, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, (T)fp.mClassWeights[0], (T)fp.mClassWeights[1], true, sampleWeight);
            return r.Mean();
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

        double GetScore(const std::vector<Utils::BatchScore> &scores)
        {
            auto score = 0.0;
            for (const auto &s : scores)
            {
                score += s.mScore;
            }
            return score / (BATCH * scores.size());
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
