#pragma once

#include "Code.h"
#include "Memory.h"
#include "../Utils/Dataset.h"

namespace SymbolicRegression::Computer
{
    template <typename T, size_t BATCH>
    struct Processor
    {
        constexpr static size_t INSTRUCTIONS_COUNT_LIMIT = 64;
        constexpr static size_t INSTRUCTIONS_COUNT{std::tuple_size<Instructions::Set>()};
        static_assert(INSTRUCTIONS_COUNT < INSTRUCTIONS_COUNT_LIMIT);

        static constexpr Instructions::Set mInstructions{};
        const CodeSettings mCodeSettings;

        explicit Processor(const CodeSettings &cs) noexcept
            : mCodeSettings(cs)
        {
        }

        template <typename INSTR>
        void Execute(INSTR &&_i,
                     const T *src1,
                     const T *src2,
                     T *__restrict dst) const noexcept
        {
            for (size_t n = 0; n < BATCH; n++)
                dst[n] = _i(src1[n], src2[n]);
        }

        template <typename INSTR>
        void Execute(INSTR &&_i,
                     const T *src1,
                     const T src2,
                     T *__restrict dst) const noexcept
        {
            for (size_t n = 0; n < BATCH; n++)
                dst[n] = _i(src1[n], src2);
        }

        template <typename INSTR>
        void Execute(INSTR &&_i,
                     const T src1,
                     const T *src2,
                     T *__restrict dst) const noexcept
        {
            for (size_t n = 0; n < BATCH; n++)
                dst[n] = _i(src1, src2[n]);
        }

        template <typename INSTR>
        void Execute(INSTR &&_i,
                     const T src1,
                     const T src2,
                     T *__restrict dst) const noexcept
        {
            for (size_t n = 0; n < BATCH; n++)
                dst[n] = _i(src1, src2);
        }

        void Execute(const Code<T> &c,
            const Utils::Dataset<T, BATCH> &data,
            Memory<T, BATCH> &mem,
            size_t batchIndex,
            bool filter = false) const noexcept
        {
            for (size_t i = 0; i < c.Size(); i++)
            {
                const auto &instr = c[i];
                if (filter && !instr.mUsed)
                    continue;

                T *__restrict dst = mem[i];

                const auto src0 = instr.mSrc[0] - mCodeSettings.CodeStart();
                const auto src1 = instr.mSrc[1] - mCodeSettings.CodeStart();

#define __handle_op(x)                                                                                                          \
    case x:                                                                                                                     \
        if constexpr ((x) < INSTRUCTIONS_COUNT)                                                                                 \
        {                                                                                                                       \
            const auto &_i = std::get<((x) < INSTRUCTIONS_COUNT ? (x) : 0)>(mInstructions);                                     \
            if (!instr.mConst[0] && !instr.mConst[1])                                                                           \
            {                                                                                                                   \
                const T *_src1 = instr.mSrc[0] < mCodeSettings.mInputSize ? data.BatchX(instr.mSrc[0], batchIndex) : mem[src0]; \
                const T *_src2 = instr.mSrc[1] < mCodeSettings.mInputSize ? data.BatchX(instr.mSrc[1], batchIndex) : mem[src1]; \
                Execute(_i, _src1, _src2, dst);                                                                                 \
            }                                                                                                                   \
            else if (instr.mConst[0] && instr.mConst[1])                                                                        \
            {                                                                                                                   \
                const T c_src1 = c.mConstants[instr.mSrc[0]];                                                                   \
                const T c_src2 = c.mConstants[instr.mSrc[1]];                                                                   \
                Execute(_i, c_src1, c_src2, dst);                                                                               \
            }                                                                                                                   \
            else if (instr.mConst[0] && !instr.mConst[1])                                                                       \
            {                                                                                                                   \
                const T *_src2 = instr.mSrc[1] < mCodeSettings.mInputSize ? data.BatchX(instr.mSrc[1], batchIndex) : mem[src1]; \
                const T c_src1 = c.mConstants[instr.mSrc[0]];                                                                   \
                Execute(_i, c_src1, _src2, dst);                                                                                \
            }                                                                                                                   \
            else if (!instr.mConst[0] && instr.mConst[1])                                                                       \
            {                                                                                                                   \
                const T *_src1 = instr.mSrc[0] < mCodeSettings.mInputSize ? data.BatchX(instr.mSrc[0], batchIndex) : mem[src0]; \
                const T c_src2 = c.mConstants[instr.mSrc[1]];                                                                   \
                Execute(_i, _src1, c_src2, dst);                                                                                \
            }                                                                                                                   \
        }                                                                                                                       \
        break

                switch (static_cast<uint32_t>(instr.mOpCode))
                {
                    __handle_op(0);
                    __handle_op(1);
                    __handle_op(2);
                    __handle_op(3);
                    __handle_op(4);
                    __handle_op(5);
                    __handle_op(6);
                    __handle_op(7);
                    __handle_op(8);
                    __handle_op(9);
                    __handle_op(10);
                    __handle_op(11);
                    __handle_op(12);
                    __handle_op(13);
                    __handle_op(14);
                    __handle_op(15);
                    __handle_op(16);
                    __handle_op(17);
                    __handle_op(18);
                    __handle_op(19);
                    __handle_op(20);
                    __handle_op(21);
                    __handle_op(22);
                    __handle_op(23);
                    __handle_op(24);
                    __handle_op(25);
                    __handle_op(26);
                    __handle_op(27);
                    __handle_op(28);
                    __handle_op(29);
                    __handle_op(30);
                    __handle_op(31);
                    __handle_op(32);
                    __handle_op(33);
                    __handle_op(34);
                    __handle_op(35);
                    __handle_op(36);
                    __handle_op(37);
                    __handle_op(38);
                    __handle_op(39);
                    __handle_op(40);
                    __handle_op(41);
                    __handle_op(42);
                    __handle_op(43);
                    __handle_op(44);
                    __handle_op(45);
                    __handle_op(46);
                    __handle_op(47);
                    __handle_op(48);
                    __handle_op(49);
                    __handle_op(50);
                    __handle_op(51);
                    __handle_op(52);
                    __handle_op(53);
                    __handle_op(54);
                    __handle_op(55);
                    __handle_op(56);
                    __handle_op(57);
                    __handle_op(58);
                    __handle_op(59);
                    __handle_op(60);
                    __handle_op(61);
                    __handle_op(62);
                    __handle_op(63);
                default:
                    break;
                }
#undef __handle_op
            }
        }
    };
}
