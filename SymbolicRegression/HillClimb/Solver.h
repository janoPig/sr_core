#pragma once

#include "Mutation.h"
#include "HillClimber.h"
#include "CodeInitializer.h"
#include "../Utils/Dataset.h"
#include "../Computer/Machine.h"
#include "../Utils/LinearRegression.h"

using namespace std::chrono;
namespace SymbolicRegression::HillClimb
{
    struct CodeInfo
    {
        double mScore;
        double mPartialScore;
        std::string mEquation;
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

        Solver(const Config &config)
            : mInitialized(false),
              mConfig(config),
              mRandom(),
              mMachine(config.mCodeSettings),
              mPopulation(config.mPopulationSize, config.mCodeSettings),
              mBestCode(config.mCodeSettings),
              mBuffX(nullptr),
              mBuffY(nullptr)
        {
            mRandom.Seed(config.mRandomSeed);
        }

        ~Solver()
        {
            Utils::AlignedFree(mBuffX);
            Utils::AlignedFree(mBuffY);
        }

        template <typename CALLBACK>
        double Fit(const Dataset &data, const FitParams &fp, CALLBACK &&callback)
        {
            if (fp.mVerbose > 1)
                printf("fit start, random engine state %zu\n", mRandom.State());

            Utils::AlignedFree(mBuffX);
            Utils::AlignedFree(mBuffY);
            mBuffX = Utils::AlignedAlloc<T>(32, data.BatchCount() * BATCH);
            mBuffY = Utils::AlignedAlloc<T>(32, data.BatchCount() * BATCH);

            const auto fitStartTime = high_resolution_clock::now();

            if (!mInitialized)
            {
                Initialize(data, fp);
            }

            if (fp.mVerbose > 1)
                callback(0, mBestCode.mScore[2]);

            const CodeMutation codeMut{fp.mConstSettings, fp.mInstrProbs, fp.mFeatProbs, mRandom};
            const ConstMutation<T> constMut{mRandom, fp.mConstSettings};

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            EvaluatedCode<T> neighbour(mConfig.mCodeSettings);

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
                    const auto score = EvalPopulation(data, fp);
                    if (fp.mVerbose > 1)
                    {
                        callback(it, score);
                        // std::cout << GetExpression(mBestCode) << std::endl;
                    }
                }

                const auto [hillclimber, selIdx] = TournamentSelection(fp.mTournament);

                auto bestCode = hillclimber->Current();
                auto bestScore = LARGE_FLOAT;
                bool find = false;

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

                        EvaluateOLS(data, neighbour, std::vector<size_t>{{hillclimber->mWorstBatch.first}}, 0);

                        if (neighbour.mScore[0] >= 1.1 * hillclimber->Current().mScore[0])
                        {
                            continue;
                        }

                        if (data.BatchCount() > 1)
                            EvaluateOLS(data, neighbour, hillclimber->mSmallSet, 1);
                        else
                            neighbour.mScore[1] = neighbour.mScore[0];

                        if (neighbour.mScore[1] < bestScore)
                        {
                            bestCode = neighbour;
                            bestScore = neighbour.mScore[1];
                            find = true;
                        }
                    }
                }

                if (find)
                {
                    if (bestCode.mScore[1] < hillclimber->Best().mScore[1] * 1.15)
                    {
                        [[maybe_unused]]const auto worstBatch = EvaluateOLS(data, bestCode, hillclimber->mBigSet, 2);
                        if (bestCode.mScore[2] < hillclimber->Best().mScore[2])
                        {
                            // bestCode.mScore[0] = worstBatch.second;
                            hillclimber->Best() = bestCode;
                            // hillclimber->mWorstBatch = worstBatch;
                        }
                    }
                    hillclimber->Current() = bestCode;
                }
            }
            return EvalPopulation(data, fp);
        }

        void Predict(Dataset &data, [[maybe_unused]] uint32_t transformation, [[maybe_unused]] T clipMin, [[maybe_unused]] T clipMax) noexcept
        {
            mMachine.ComputeOLS(data, mBestCode.mCode, mBestCode.mCoeffs[0], mBestCode.mCoeffs[1]);
        }

        void Predict(Dataset &data, [[maybe_unused]] uint32_t transformation, uint32_t id, [[maybe_unused]] T clipMin, [[maybe_unused]] T clipMax) noexcept
        {
            auto &hc = mPopulation[id];
            mMachine.ComputeOLS(data, hc.Best().mCode, hc.Best().mCoeffs[0], hc.Best().mCoeffs[1]);
        }

        const auto &GetBestCode()
        {
            return mBestCode;
        }

        double EvalPopulation(const Dataset &data, [[maybe_unused]]const FitParams &fp, double alpha = 0.05) noexcept
        {
            auto bestScore = mBestCode.mScore[3];
            for (auto &hc : mPopulation)
            {
                if (hc.Best().mScore[2] > (1.0 + alpha) * bestScore)
                    continue;
                hc.Best().mScore[3] = EvaluateOLS(data, hc.Best(), mFullSet, 3);
                if (hc.Best().mScore[3] < bestScore)
                {
                    bestScore = hc.Best().mScore[3];
                    mBestCode = hc.Best();
                }
            }
            return mBestCode.mScore[3];
        }

        double Score() const noexcept
        {
            return mBestCode.mScore[3];
        }

        CodeInfo GetBestInfo() noexcept
        {
            return CodeInfo{mBestCode.mScore[3], mBestCode.mScore[2], GetExpression(mBestCode)};
        }

        CodeInfo GetInfo(size_t idx) noexcept
        {
            auto &code = mPopulation[idx].Best();
            return CodeInfo{code.mScore[3], code.mScore[2], GetExpression(code)};
        }

        std::string GetExpression(EvaluatedCode<T> &c) noexcept
        {
            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            c.mCode.IsConstExpression(indices.data(), mCodeMapping.set);
            auto str = c.mCode.GetString(mCodeMapping.set);
            std::ostringstream out;

            out << c.mCoeffs[0] << "+" << c.mCoeffs[1] << "*" << str;
            return out.str();
        }

        const Config &GetConfig() const noexcept
        {
            return mConfig;
        }

    private:
        void Initialize(const Dataset &data, const FitParams &fp)
        {
            assert(!mInitialized);

            const CodeInitializer<T> codeInit{mConfig.mCodeSettings, mConfig.mInitConstSettings, fp.mInstrProbs, fp.mFeatProbs, mRandom};
            auto smallSize = std::min((size_t)8, data.BatchCount());
            auto bigSize = std::min((size_t)64, data.BatchCount());

            mFullSet.resize(data.BatchCount());
            std::iota(mFullSet.begin(), mFullSet.end(), 0);

            std::vector<uint32_t> indices;
            indices.reserve((size_t)mConfig.mCodeSettings.mMaxCodeSize * 2);

            std::vector<size_t> bs(1);
            bs[0] = mRandom.Rand(data.BatchCount());

            for (auto &hc : mPopulation)
            {
                hc.mSmallSet.resize(smallSize);
                for (auto &batch : hc.mSmallSet)
                {
                    batch = mRandom.Rand(data.BatchCount());
                }

                hc.mBigSet.resize(bigSize);
                for (auto &batch : hc.mBigSet)
                {
                    batch = mRandom.Rand(data.BatchCount());
                }

                EvaluatedCode<T> candidate{mConfig.mCodeSettings};

                int cnt = 3;
                int k = 30;
                auto bestScore = LARGE_FLOAT;
                while (cnt && k)
                {
                    codeInit.operator()(candidate.mCode);

                    if (!candidate.mCode.IsConstExpression(indices.data(), mCodeMapping.set))
                    {
                        Evaluate(data, candidate, bs, 1, fp);

                        if (cnt == 3 || candidate.mScore[0] < bestScore)
                        {
                            bestScore = candidate.mScore[0];
                            hc.Current() = candidate;
                            cnt--;
                        }
                    }
                    k--;
                }
                Evaluate(data, hc.Current(), hc.mSmallSet, 1, fp);
                hc.mWorstBatch = Evaluate(data, hc.Current(), hc.mBigSet, 2, fp);
                hc.Best() = hc.Current();

                if (hc.Best().mScore[2] < mBestCode.mScore[2])
                {
                    mBestCode = hc.Current();
                }
            }

            mInitialized = true;
        }

        auto Evaluate(const Dataset &data, EvaluatedCode<T> &evc, const std::vector<size_t> &batchSelection, int id, const FitParams &fp) noexcept
        {
            Utils::Result r;
            auto x = mMachine.ComputeScore(data, evc.mCode, batchSelection, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, true);
            evc.mScore[id] = r.mean();
            return x;
        }

        auto EvaluateOLS(const Dataset &data, EvaluatedCode<T> &evc, const std::vector<size_t> &batchSelection, int id) noexcept
        {
            auto [err, B0, B1] = mMachine.ComputeScoreOLS(data, evc.mCode, batchSelection, mBuffX, mBuffY);
            evc.mScore[id] = err;
            evc.mCoeffs[0] = B0;
            evc.mCoeffs[1] = B1;
            return err;
        }

        auto Evaluate(const Dataset &data, const EvaluatedCode<T> &evc, const FitParams &fp) noexcept
        {
            Utils::Result r;
            [[maybe_unused]] const auto x = mMachine.ComputeScore(data, evc.mCode, mFullSet, r, mConfig.mTransformation, fp.mMetric, (T)mConfig.mClipMin, (T)mConfig.mClipMax, true);
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
                if (tmp.Best().mScore[2] < bestFit)
                {
                    bestFit = tmp.Best().mScore[2];
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
        EvaluatedCode<T> mBestCode;

        std::vector<size_t> mFullSet;

        T *__restrict mBuffX;
        T *__restrict mBuffY;
    };
}