#pragma once

#include "../Defs.h"
#include "../Computer/Code.h"

namespace SymbolicRegression
{
    template <typename T>
    struct EvaluatedCode
    {
        using Code = Computer::Code<T>;

        EvaluatedCode() = default;

        explicit EvaluatedCode(const CodeSettings &cs) noexcept
            : mCode(cs)
        {
        }

        void ResetScore() noexcept
        {
            for (size_t i = 0; i < std::size(mScore); i++)
            {
                mScore[i] = LARGE_FLOAT;
            }
        }

        Code mCode{};
        double mScore[4] = {LARGE_FLOAT, LARGE_FLOAT, LARGE_FLOAT, LARGE_FLOAT};
        T mCoeffs[2];
    };
}
