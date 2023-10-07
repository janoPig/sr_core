#pragma once

#include "../Computer/Code.h"

namespace SymbolicRegression
{
    using Computer::Instruction;

    template <typename T>
    struct CodeInitializer
    {
        void operator()(Computer::Code<T> &c) const noexcept
        {
            auto newSrc = [this, &c](Instruction &instr, uint32_t j, uint32_t I) noexcept
            {
                const auto constCount = (uint32_t)c.mConstants.size();

                if (j == 0 || mRandom.TestProb(512))
                {
                    if (constCount == 0 || mRandom.TestProb(768))
                    {
                        instr.mSrc[I] = mFeatProbs(mRandom);
                        instr.mConst[I] = false;
                    }
                    else
                    {
                        instr.mSrc[I] = mRandom.Rand(constCount);
                        instr.mConst[I] = true;
                    }
                    return;
                }
                else
                {
                    instr.mSrc[I] = mRandom.Rand((uint32_t)j) + c.CodeStart();
                    instr.mConst[I] = false;
                }
            };

            uint32_t size = mCodeSettings.mMaxCodeSize;
            if (mCodeSettings.mMinCodeSize < mCodeSettings.mMaxCodeSize)
            {
                size = mRandom.Rand(mCodeSettings.mMinCodeSize, mCodeSettings.mMaxCodeSize + 1);
            }

            c.SetSize(size);

            for (size_t i = 0; i < c.Size(); i++)
            {
                newSrc(c[i], (uint32_t)i, 0);
                newSrc(c[i], (uint32_t)i, 1);
                c[i].mOpCode = mInstrProbs(mRandom);
            }

            const auto predef = mConstSettings.mPredefinedProb > 0 && !mConstSettings.mPredefinedSet.empty();

            for (size_t i = 0; i < c.mConstants.size(); i++)
            {
                if (predef && (mConstSettings.mPredefinedProb == 1.0 || mRandom.Rand(1.0) < mConstSettings.mPredefinedProb))
                {
                    c.mConstants[i] = (T)mRandom.RandomElement(mConstSettings.mPredefinedSet);
                }
                else
                {
                    c.mConstants[i] = static_cast<T>(mRandom.Rand(mConstSettings.mMin, mConstSettings.mMax));
                }
            }
        }

        const CodeSettings &mCodeSettings;
        const ConstSettings &mConstSettings;
        const Utils::DiscreteRandomVariable<Computer::Instructions::InstructionID> mInstrProbs;
        const Utils::DiscreteRandomVariable<uint32_t> mFeatProbs;
        Utils::RandomEngine &mRandom;
    };
}