#pragma once
// https://commons.wikimedia.org/wiki/Fuzzy_operator#Dyadic_Operators_based_on_a_Hyperbolic_Paraboloid

namespace SymbolicRegression::Computer::Instructions
{
    DEF_INSTR(f_and, 2, a *b);
    DEF_INSTR(f_or, 2, a + b - a * b);
    DEF_INSTR(f_xor, 2, a + b - static_cast<T>(2.0) * a * b);
    DEF_INSTR(f_impl, 2, static_cast<T>(1.0) - a + a * b);
    DEF_INSTR(f_not, 1, static_cast<T>(1.0) - a);
    DEF_INSTR(f_nand, 2, static_cast<T>(1.0) - a * b);
    DEF_INSTR(f_nor, 2, static_cast<T>(1.0) - a - b + a * b);
    DEF_INSTR(f_nxor, 2, static_cast<T>(1.0) - a - b + static_cast<T>(2.0) * a * b);
    DEF_INSTR(f_nimpl, 2, a *(static_cast<T>(1.0) - b));
}