#pragma once

namespace SymbolicRegression::Computer::Instructions
{
    enum class InstructionID : uint32_t
    {
        nop = 0,
        add,
        sub,
        mul,
        div,
        inv,
        minv,
        sq2,
        pdiv,

        max,
        min,
        abs,
        floor,
        ceil,
        lt,
        gt,
        lte,
        gte,

        pow,
        exp,
        log,
        sqrt,
        cbrt,
        aq,

        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,

        f_and,
        f_or,
        f_xor,
        f_impl,
        f_not,
        f_nand,
        f_nor,
        f_nxor,
        f_nimpl
    };

#define DEF_INSTR(name, op, code)                                                                     \
    struct instruction_##name                                                                         \
    {                                                                                                 \
        constexpr static uint32_t operands = op;                                                      \
        constexpr static InstructionID id = InstructionID::name;                                      \
        template <typename T>                                                                         \
        constexpr T operator()([[maybe_unused]] const T a, [[maybe_unused]] const T b) const noexcept \
        {                                                                                             \
            (void)b;                                                                                  \
            return (code);                                                                            \
        }                                                                                             \
        constexpr auto get_name() const noexcept                                                      \
        {                                                                                             \
            return #name;                                                                             \
        }                                                                                             \
        constexpr auto get_code() const noexcept                                                      \
        {                                                                                             \
            return #code;                                                                             \
        }                                                                                             \
    }
}
