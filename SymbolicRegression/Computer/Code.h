#pragma once

#include "../Utils/Utils.h"
#include "../Config.h"
#include "./Instructions/Instructions.h"

namespace SymbolicRegression::Computer
{
    struct Instruction
    {
        Instructions::InstructionID mOpCode{Instructions::InstructionID::nop};
        uint32_t mSrc[2]{0, 0};
        bool mConst[2]{false, false};
        bool mUsed{false};
    };

    template <typename T>
    struct Code
    {
        Code() = default;
        explicit Code(const CodeSettings &cs) noexcept
            : mInputSize(cs.mInputSize),
              mCodeSize(0),
              mConstants(cs.mConstSize),
              mCodeInstructions(cs.mMaxCodeSize),
              mUsedInstructionsCount(0),
              mUsedInstructions(0),
              mUsedConst(cs.mConstSize)
        {
        }

        auto MaxSize() const noexcept
        {
            return mCodeInstructions.size();
        }

        auto Size() const noexcept
        {
            return mCodeSize;
        }

        void SetSize(uint32_t size) noexcept
        {
            assert(size <= mCodeInstructions.size());
            mCodeSize = size;
        }

        T *Constants() noexcept
        {
            return mConstants.data();
        }

        const Instruction &operator[](size_t idx) const noexcept
        {
            assert(idx < mCodeSize);
            return mCodeInstructions[idx];
        }

        Instruction &operator[](size_t idx) noexcept
        {
            assert(idx < mCodeSize);
            return mCodeInstructions[idx];
        }

        auto CodeStart() const noexcept
        {
            return mInputSize;
        }

        bool IsConstExpression(uint32_t *const indices, const std::vector<CodeGen::InstructionInfo> &set) noexcept
        {
            bool ret = true;
            int32_t ic = 0;
            mUsedInstructions.clear();
            mUsedConst.clear();
            auto &first = mCodeInstructions[mCodeSize - 1];
            const auto codeStart = CodeStart();

            const auto process_instr = [&](const auto &instr, const uint32_t I)
            {
                if (instr.mConst[I])
                {
                    const auto it = std::find(mUsedConst.begin(), mUsedConst.end(), instr.mSrc[I]);
                    if (it == mUsedConst.end())
                    {
                        mUsedConst.push_back(instr.mSrc[I]);
                    }
                }
                else
                {
                    ret = false;
                    if (instr.mSrc[I] >= codeStart)
                        indices[ic++] = instr.mSrc[I];
                }
            };

            process_instr(first, 0);
            if (set[static_cast<uint32_t>(first.mOpCode)].op > 1)
                process_instr(first, 1);

            for (auto &i : mCodeInstructions)
                i.mUsed = false;

            first.mUsed = true;
            mUsedInstructionsCount = 1;
            mUsedInstructions.push_back(mCodeSize - 1);

            while (ic)
            {
                const auto i = indices[--ic];
                auto &instr = mCodeInstructions[i - codeStart];
                if (!instr.mUsed)
                {
                    mUsedInstructions.push_back(i - codeStart);
                    mUsedInstructionsCount++;
                    instr.mUsed = true;
                }

                process_instr(instr, 0);
                if (set[static_cast<uint32_t>(instr.mOpCode)].op > 1)
                    process_instr(instr, 1);
            }
            return ret;
        }

        auto GetString(const std::vector<CodeGen::InstructionInfo> &set) const noexcept
        {
            std::vector<std::string> tmp;
            tmp.resize(mCodeSize);

            auto parse = [&](size_t idx, bool isConst) -> std::string
            {
                if (isConst)
                    return "(" + Utils::ToStringWithPrecision(mConstants[(uint32_t)idx]) + ")";
                else if (idx < mInputSize)
                    return "x" + std::to_string(idx + 1);
                else
                    return tmp[idx - CodeStart()];
            };

            for (size_t i = 0; i < mCodeSize; i++)
            {
                const auto &instr = mCodeInstructions[i];
                if (!instr.mUsed)
                    continue;
                const auto &info = set[static_cast<uint32_t>(instr.mOpCode)];

                tmp[i] = info.name;
                std::string _op = ",";
                std::string _uop = "";
                if (info.name == "add")
                {
                    _op = "+";
                    tmp[i] = "";
                }
                if (info.name == "mul")
                {
                    _op = "*";
                    tmp[i] = "";
                }
                if (info.name == "div")
                {
                    _op = "/";
                    tmp[i] = "";
                }
                if (info.name == "sub")
                {
                    _op = "-";
                    tmp[i] = "";
                }
                if (info.name == "lt")
                {
                    _op = "<";
                    tmp[i] = "";
                }
                if (info.name == "gt")
                {
                    _op = ">";
                    tmp[i] = "";
                }
                if (info.name == "lte")
                {
                    _op = "<=";
                    tmp[i] = "";
                }
                if (info.name == "gte")
                {
                    _op = ">=";
                    tmp[i] = "";
                }
                if (info.name == "inv")
                {
                    _uop = "-";
                    tmp[i] = "";
                }
                if (info.name == "minv")
                {
                    _uop = "1.0/";
                    tmp[i] = "";
                }
                if (info.name == "sq2")
                {
                    tmp[i] = "";
                }
                if (info.name == "nop")
                {
                    tmp[i] = "";
                }
                if (info.name == "f_and")
                {
                    _op = "&";
                    tmp[i] = "";
                }
                if (info.name == "f_or")
                {
                    _op = "|";
                    tmp[i] = "";
                }
                if (info.name == "f_xor")
                {
                    _op = "^";
                    tmp[i] = "";
                }
                if (info.name == "f_not")
                {
                    _uop = "~";
                    tmp[i] = "";
                }

                if (info.op > 0)
                    tmp[i] += "(" + _uop + parse(instr.mSrc[0], instr.mConst[0]);

                if (info.op > 1)
                {
                    tmp[i] += _op + parse(instr.mSrc[1], instr.mConst[1]);
                }
                if (info.name == "sq2")
                {
                    tmp[i] += "**2";
                }
                if (info.op > 0)
                    tmp[i] += ")";
            }

            return tmp[mCodeSize - 1];
        }

        uint32_t mInputSize{};
        uint32_t mCodeSize{};
        std::vector<T> mConstants{};
        std::vector<Instruction> mCodeInstructions{};

        size_t mUsedInstructionsCount{};
        std::vector<uint32_t> mUsedInstructions{};
        std::vector<uint32_t> mUsedConst{};
    };
}
