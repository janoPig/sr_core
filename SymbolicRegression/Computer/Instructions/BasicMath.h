#pragma once

namespace SymbolicRegression::Computer::Instructions
{
    DEF_INSTR(nop, 1, a);
    DEF_INSTR(add, 2, a + b);
    DEF_INSTR(sub, 2, a - b);
    DEF_INSTR(mul, 2, a *b);
    DEF_INSTR(div, 2, a / b);
    DEF_INSTR(inv, 1, -a);
    DEF_INSTR(minv, 1, 1.0f / a);
    DEF_INSTR(sq2, 1, a *a);
    DEF_INSTR(max, 2, a > b ? a : b);
    DEF_INSTR(min, 2, a < b ? a : b);
    DEF_INSTR(abs, 1, a < 0 ? -a : a);
    DEF_INSTR(floor, 1, std::floor(a));
    DEF_INSTR(ceil, 1, std::ceil(a));
    DEF_INSTR(lt, 2, static_cast<T>(a < b));
    DEF_INSTR(gt, 2, static_cast<T>(a > b));
    DEF_INSTR(lte, 2, static_cast<T>(a <= b));
    DEF_INSTR(gte, 2, static_cast<T>(a >= b));
}