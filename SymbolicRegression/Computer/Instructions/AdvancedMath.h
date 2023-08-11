#pragma once

namespace SymbolicRegression::Computer::Instructions
{
    DEF_INSTR(pow, 2, std::pow(a, b));
    DEF_INSTR(exp, 1, std::exp(a));
    DEF_INSTR(log, 1, std::log(a));
    DEF_INSTR(sqrt, 1, std::sqrt(a));
    DEF_INSTR(cbrt, 1, std::cbrt(a));
    DEF_INSTR(aq, 2, a / std::sqrt(static_cast<T>(1.0) + b * b));
    DEF_INSTR(sin, 1, std::sin(a));
    DEF_INSTR(cos, 1, std::cos(a));
    DEF_INSTR(tan, 1, std::tan(a));
    DEF_INSTR(asin, 1, std::asin(a));
    DEF_INSTR(acos, 1, std::acos(a));
    DEF_INSTR(atan, 1, std::atan(a));
    DEF_INSTR(sinh, 1, std::sinh(a));
    DEF_INSTR(cosh, 1, std::cosh(a));
    DEF_INSTR(tanh, 1, std::tanh(a));
    DEF_INSTR(pdiv, 2, a / std::sqrt(static_cast<T>(0.00000001) + b * b));
}