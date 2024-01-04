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
              mTreeComplexity(0),
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
            mTreeComplexity = 1;
            mUsedInstructions.push_back(mCodeSize - 1);

            while (ic)
            {
                const auto i = indices[--ic];
                auto &instr = mCodeInstructions[i - codeStart];
                if (!instr.mUsed)
                {
                    mUsedInstructions.push_back(i - codeStart);
                    instr.mUsed = true;
                }
                mTreeComplexity++;

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
            const auto ci = GetConstIndices();

            auto parse = [&](uint32_t idx, bool isConst) -> std::string
            {
                if (isConst)
                {
                    if (auto it = ci.find(idx); it != ci.end())
                    {
                        return "c" + std::to_string(it->second);
                    }
                    else
                    {
                        return "error";
                    }
                }
                else if (idx < mInputSize)
                    return "x" + std::to_string(idx);
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

        auto GetConstants() const noexcept
        {
            if (!mUsedConst.empty())
            {
                std::vector<double> result(mUsedConst.size());
                for (size_t i = 0; i < mUsedConst.size(); i++)
                {
                    result[i] = static_cast<double>(mConstants[mUsedConst[i]]);
                }
                return result;
            }
            return std::vector<double>{};
        }

        auto GenerateCode(const std::vector<CodeGen::InstructionInfo> &set, const std::string &funcName) const
        {
            const auto ci = GetConstIndices();
            std::string result = "def " + funcName + "(x, c):\n";
            for (size_t i = 0; i < mCodeSize; i++)
            {
                const auto &instr = mCodeInstructions[i];
                if (!instr.mUsed)
                    continue;
                const auto &info = set[static_cast<uint32_t>(instr.mOpCode)];

                auto parse = [&](uint32_t idx) -> std::string
                {
                    if (instr.mConst[idx])
                    {
                        if (auto it = ci.find(instr.mSrc[idx]); it != ci.end())
                        {
                            return "c[" + std::to_string(it->second) + "]";
                        }
                        else
                        {
                            return "error";
                        }
                    }
                    else if (instr.mSrc[idx] < mInputSize)
                        return "x[:, " + std::to_string(instr.mSrc[idx]) + "]";
                    else
                        return "tmp_" + std::to_string(instr.mSrc[idx] - CodeStart());
                };

#define HANDLE_SYMBOL(iid, fn)                                     \
    case iid:                                                      \
        result += "\ttmp_" + std::to_string(i) + " = " + fn + "("; \
        if (info.op > 0)                                           \
            result += parse(0);                                    \
        if (info.op > 1)                                           \
            result += ", " + parse(1);                             \
        result += ")\n";                                           \
        break

#define HANDLE_INEQUALITY(iid, code)                                                              \
    case iid:                                                                                     \
        result += "\ttmp_" + std::to_string(i) + " = (" + parse(0) + code + parse(1) + ")*1.0\n"; \
        break

                switch (instr.mOpCode)
                {
                    HANDLE_SYMBOL(Instructions::InstructionID::nop, "");
                    HANDLE_SYMBOL(Instructions::InstructionID::add, "numpy.add");
                    HANDLE_SYMBOL(Instructions::InstructionID::sub, "numpy.subtract");
                    HANDLE_SYMBOL(Instructions::InstructionID::mul, "numpy.multiply");
                    HANDLE_SYMBOL(Instructions::InstructionID::div, "numpy.divide");
                    HANDLE_SYMBOL(Instructions::InstructionID::inv, "numpy.negative");
                    HANDLE_SYMBOL(Instructions::InstructionID::minv, "1.0/");
                    HANDLE_SYMBOL(Instructions::InstructionID::sq2, "numpy.square");
                case Instructions::InstructionID::pdiv:
                    result += "\ttmp_" + std::to_string(i) + " = " + parse(0) + " / (numpy.sqrt(0.00000001 + numpy.square(" + parse(1) + "))) # PDIV\n";
                    break;
                    HANDLE_SYMBOL(Instructions::InstructionID::max, "numpy.maximum");
                    HANDLE_SYMBOL(Instructions::InstructionID::min, "numpy.minimum");
                    HANDLE_SYMBOL(Instructions::InstructionID::abs, "numpy.absolute");
                    HANDLE_SYMBOL(Instructions::InstructionID::floor, "numpy.floor");
                    HANDLE_SYMBOL(Instructions::InstructionID::ceil, "numpy.ceil");
                    HANDLE_INEQUALITY(Instructions::InstructionID::lt, " < ");
                    HANDLE_INEQUALITY(Instructions::InstructionID::gt, " > ");
                    HANDLE_INEQUALITY(Instructions::InstructionID::lte, " <= ");
                    HANDLE_INEQUALITY(Instructions::InstructionID::gte, " >= ");
                    HANDLE_SYMBOL(Instructions::InstructionID::pow, "numpy.power");
                    HANDLE_SYMBOL(Instructions::InstructionID::exp, "numpy.exp");
                    HANDLE_SYMBOL(Instructions::InstructionID::log, "numpy.log");
                    HANDLE_SYMBOL(Instructions::InstructionID::sqrt, "numpy.sqrt");
                    HANDLE_SYMBOL(Instructions::InstructionID::cbrt, "numpy.cbrt");
                case Instructions::InstructionID::aq:
                    result += "\ttmp_" + std::to_string(i) + " = " + parse(0) + " / (numpy.sqrt(1.0 + numpy.square(" + parse(1) + "))) # AQ\n";
                    break;
                    HANDLE_SYMBOL(Instructions::InstructionID::sin, "numpy.sin");
                    HANDLE_SYMBOL(Instructions::InstructionID::cos, "numpy.cos");
                    HANDLE_SYMBOL(Instructions::InstructionID::tan, "numpy.tan");
                    HANDLE_SYMBOL(Instructions::InstructionID::asin, "numpy.arcsin");
                    HANDLE_SYMBOL(Instructions::InstructionID::acos, "numpy.arccos");
                    HANDLE_SYMBOL(Instructions::InstructionID::atan, "numpy.arctan");
                    HANDLE_SYMBOL(Instructions::InstructionID::sinh, "numpy.sinh");
                    HANDLE_SYMBOL(Instructions::InstructionID::cosh, "numpy.cosh");
                    HANDLE_SYMBOL(Instructions::InstructionID::tanh, "numpy.tanh");
                case Instructions::InstructionID::f_and:
                    result += "\ttmp_" + std::to_string(i) + " = " + parse(0) + " * " + parse(1) + " # F_AND\n";
                    break;
                case Instructions::InstructionID::f_or:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = " + a + " + " + b + " - " + a + " * " + b + " # F_OR\n";
                }
                break;
                case Instructions::InstructionID::f_xor:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = " + a + " + " + b + " - 2.0 * " + a + " * " + b + " # F_XOR\n";
                }
                break;
                case Instructions::InstructionID::f_impl:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = 1.0 - " + a + " + " + a + " * " + b + " # F_IMPL\n";
                }
                break;
                case Instructions::InstructionID::f_not:
                    result += "\ttmp_" + std::to_string(i) + " = 1.0 - " + parse(0) + " # F_NOT\n";
                    break;
                case Instructions::InstructionID::f_nand:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = 1.0 - " + a + " * " + b + " # F_NAND\n";
                }
                break;
                case Instructions::InstructionID::f_nor:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = 1.0 - " + a + " - " + b + " + " + a + " * " + b + " # F_NOR\n";
                }
                break;
                case Instructions::InstructionID::f_nxor:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = 1.0 - " + a + " - " + b + " + 2.0 * " + a + " * " + b + " # F_NXOR\n";
                }
                break;
                case Instructions::InstructionID::f_nimpl:
                {
                    const auto a = parse(0);
                    const auto b = parse(1);
                    result += "\ttmp_" + std::to_string(i) + " = " + a + " * (1.0 -" + b + ") # F_NIMPL\n";
                }
                break;
                default:
                    break;
                }
            }

            result += std::string("\treturn ") + "tmp_" + std::to_string(mCodeSize - 1) + "\n";

            return result;
        }

        auto GetConstIndices() const noexcept
        {
            std::unordered_map<uint32_t, uint32_t> ci{};
            uint32_t i = 0;
            std::transform(mUsedConst.begin(), mUsedConst.end(), std::inserter(ci, ci.end()),
                           [&i](const int &s)
                           { return std::make_pair(s, i++); });

            return ci;
        }

        uint32_t mInputSize{};
        uint32_t mCodeSize{};
        std::vector<T> mConstants{};
        std::vector<Instruction> mCodeInstructions{};

        size_t mTreeComplexity{};
        std::vector<uint32_t> mUsedInstructions{};
        std::vector<uint32_t> mUsedConst{};
    };
}
