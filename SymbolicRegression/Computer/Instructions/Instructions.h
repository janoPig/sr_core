#pragma once

#include "Prototype.h"
#include "BasicMath.h"
#include "AdvancedMath.h"
#include "Fuzzy.h"
#include "CodeGen.h"

namespace SymbolicRegression::Computer::Instructions
{
    using Set = std::tuple<
        instruction_nop,
        instruction_add,
        instruction_sub,
        instruction_mul,
        instruction_div,
        instruction_inv,
        instruction_minv,
        instruction_sq2,
        instruction_pdiv,

        instruction_max,
        instruction_min,
        instruction_abs,
        instruction_floor,
        instruction_ceil,
        instruction_lt,
        instruction_gt,
        instruction_lte,
        instruction_gte,

        instruction_pow,
        instruction_exp,
        instruction_log,
        instruction_sqrt,
        instruction_cbrt,
        instruction_aq,

        instruction_sin,
        instruction_cos,
        instruction_tan,
        instruction_asin,
        instruction_acos,
        instruction_atan,
        instruction_sinh,
        instruction_cosh,
        instruction_tanh,

        instruction_f_and,
        instruction_f_or,
        instruction_f_xor,
        instruction_f_impl,
        instruction_f_not,
        instruction_f_nand,
        instruction_f_nor,
        instruction_f_nxor,
        instruction_f_nimpl>;

    inline const std::vector<std::pair<InstructionID, double>> BasicMath = {{{InstructionID::nop, 0.01},
                                                                             {InstructionID::add, 1.0},
                                                                             {InstructionID::sub, 1.0},
                                                                             {InstructionID::mul, 1.0},
                                                                             {InstructionID::div, 0.1},
                                                                             {InstructionID::sq2, 0.05}}};

    inline const std::vector<std::pair<InstructionID, double>> AdvancedMath = {{{InstructionID::nop, 0.01},
                                                                                {InstructionID::add, 1.0},
                                                                                {InstructionID::sub, 1.0},
                                                                                {InstructionID::mul, 1.0},
                                                                                {InstructionID::div, 0.1},
                                                                                {InstructionID::sq2, 0.05},
                                                                                {InstructionID::pow, 0.001},
                                                                                {InstructionID::exp, 0.001},
                                                                                {InstructionID::log, 0.001},
                                                                                {InstructionID::sqrt, 0.1},
                                                                                {InstructionID::sin, 0.005},
                                                                                {InstructionID::cos, 0.005},
                                                                                {InstructionID::tan, 0.001},
                                                                                {InstructionID::asin, 0.001},
                                                                                {InstructionID::acos, 0.001},
                                                                                {InstructionID::atan, 0.001},
                                                                                {InstructionID::sinh, 0.001},
                                                                                {InstructionID::cosh, 0.001},
                                                                                {InstructionID::tanh, 0.001}}};

    inline const std::vector<std::pair<InstructionID, double>> FuzzyMath = {{{InstructionID::nop, 0.01},
                                                                             {InstructionID::f_and, 1.0},
                                                                             {InstructionID::f_or, 1.0},
                                                                             {InstructionID::f_xor, 1.0},
                                                                             {InstructionID::f_not, 1.0}}};
}
