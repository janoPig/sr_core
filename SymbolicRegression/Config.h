#pragma once

#include "Computer/Instructions/Prototype.h"

namespace SymbolicRegression
{
    struct CodeSettings
    {
        uint32_t mInputSize;
        uint32_t mConstSize;
        uint32_t mMinCodeSize;
        uint32_t mMaxCodeSize;

        constexpr auto MaxMemorySize() const noexcept
        {
            return mInputSize + mMaxCodeSize;
        }

        constexpr auto CodeStart() const noexcept
        {
            return mInputSize;
        }
    };

    struct ConstSettings
    {
        double mMin;
        double mMax;
        double mPredefinedProb;
        std::vector<double> mPredefinedSet;
    };

    struct Config
    {
        uint64_t mRandomSeed;
        uint32_t mPopulationSize;
        uint32_t mTransformation;
        double mClipMin;
        double mClipMax;
        ConstSettings mInitConstSettings;
        CodeSettings mCodeSettings;
    };

    struct FitParams
    {
        uint32_t mTimeLimit;
        uint32_t mVerbose;
        uint32_t mTournament;
        uint32_t mMetric;
        uint64_t mIterLimit;
        ConstSettings mConstSettings;
        std::vector<std::pair<Computer::Instructions::InstructionID, double>> mInstrProbs;
        std::vector<std::pair<uint32_t, double>> mFeatProbs;
    };
}
