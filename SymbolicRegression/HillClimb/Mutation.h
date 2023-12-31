#pragma once

#include "../Utils/Rand.h"
#include "../Computer/Code.h"

namespace SymbolicRegression::HillClimb
{
    template <typename T>
    T mutate_const_4(Utils::RandomEngine &random, T val, T clipMin, T clipMax, T factor = static_cast<T>(1.0)) noexcept
    {
        constexpr auto epsilon = static_cast<T>(0.000001);
        // volatile: msvc incorrectly optimize this
        volatile auto delta = random.Rand(static_cast<T>(0.0), static_cast<T>(1.0 - epsilon));
        delta = delta * delta * delta * delta * factor;
        delta += static_cast<T>(epsilon);
        if (random.Rand())
            val *= static_cast<T>(1.0) + delta;
        else
            val /= static_cast<T>(1.0) + delta;

        if (val > clipMax)
            val = clipMax;
        else if (val < clipMin)
            val = clipMin;
        return val;
    }

    struct CodeMutation
    {
        template <typename T>
        void operator()(Computer::Code<T> &code) const noexcept
        {
            const auto instrPos = code.mUsedInstructions[mRandom.Rand(code.mUsedInstructions.size())];
            assert(code[instrPos].mUsed);

            MuteAtPos(code, instrPos);
            const auto &instr = code.mCodeInstructions[instrPos];
            if (!instr.mConst[0] && instr.mSrc[0] >= code.CodeStart())
            {
                if (mRandom.TestProb(512))
                {
                    MuteAtPos(code, instr.mSrc[0] - code.CodeStart());
                }
            }
            if (!instr.mConst[1] && instr.mSrc[1] >= code.CodeStart())
            {
                if (mRandom.TestProb(512))
                {
                    MuteAtPos(code, instr.mSrc[1] - code.CodeStart());
                }
            }
        }

        template <typename T>
        void NewSrc(Computer::Code<T> &code, uint32_t instrPos, uint32_t I) const noexcept
        {
            const auto constCount = (uint32_t)code.mConstants.size();
            auto &instr = code.mCodeInstructions[instrPos];

            // mute vals
            if (instrPos == 0 || mRandom.TestProb(512))
            {
                if (constCount == 0 || mRandom.TestProb(768))
                {
                    instr.mSrc[I] = mFeatProbs(mRandom);
                    instr.mConst[I] = false;
                }
                else
                {
                    // mute to const
                    const auto ncp = mRandom.Rand(constCount);

                    if (instr.mConst[I])
                    {
                        if (mRandom.TestProb(512)) // copy constant to new pos
                            code.mConstants[ncp] = code.mConstants[instr.mSrc[I]];
                    }

                    code.mConstants[ncp] = mutate_const_4(mRandom, code.mConstants[ncp], static_cast<T>(mConstSettings.mMin), static_cast<T>(mConstSettings.mMax), static_cast<T>(0.1));

                    instr.mSrc[I] = ncp;
                    instr.mConst[I] = true;
                }
            }
            else
            {
                instr.mSrc[I] = mRandom.Rand((uint32_t)instrPos) + code.CodeStart();
                instr.mConst[I] = false;
            }
        }

        template <typename T>
        void MuteAtPos(Computer::Code<T> &code, uint32_t instrPos) const noexcept
        {
            auto &instr = code.mCodeInstructions[instrPos];

            if (mRandom.TestProb(128))
            {
                std::swap(instr.mSrc[0], instr.mSrc[1]);
                std::swap(instr.mConst[0], instr.mConst[1]);
            }

            if (mRandom.TestProb(256))
            {
                NewSrc(code, instrPos, 1);
            }

            if (mRandom.TestProb(256))
            {
                NewSrc(code, instrPos, 0);
            }
            else
            {
                instr.mOpCode = mInstrProbs(mRandom);
            }
        }

        const ConstSettings &mConstSettings;
        Utils::DiscreteRandomVariable<Computer::Instructions::InstructionID> mInstrProbs;
        Utils::DiscreteRandomVariable<uint32_t> mFeatProbs;
        Utils::RandomEngine &mRandom;
    };

    template <typename T>
    struct ConstMutation
    {
        using Code = Computer::Code<T>;

        ConstMutation(Utils::RandomEngine &random, const ConstSettings &constSettings) noexcept
            : mRandom(random), mConstSettings(constSettings), mUsePredef(constSettings.mPredefinedProb > 0.0 && !constSettings.mPredefinedSet.empty())
        {
        }

        void operator()(Code &code) const noexcept
        {
            if (code.mUsedConst.size() == 0)
                return;

            auto pos = code.mUsedConst[mRandom.Rand(code.mUsedConst.size())];

            if (mUsePredef && (mConstSettings.mPredefinedProb == 1.0 || mRandom.Rand(1.0) < mConstSettings.mPredefinedProb))
            {
                code.mConstants[pos] = (T)mRandom.RandomElement(mConstSettings.mPredefinedSet);
            }
            else
            {
                code.mConstants[pos] = mutate_const_4(mRandom, code.mConstants[pos], static_cast<T>(mConstSettings.mMin), static_cast<T>(mConstSettings.mMax), static_cast<T>(1.0));
            }
        }

        Utils::RandomEngine &mRandom;
        const ConstSettings &mConstSettings;
        const bool mUsePredef;
    };
}
