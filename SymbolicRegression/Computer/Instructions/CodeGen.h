#pragma once

namespace SymbolicRegression::Computer::CodeGen
{
    template <typename instruction>
    std::string generate_function(const instruction &instr)
    {
        std::string ret = "template<typename T>\n";
        ret += "inline T instruction_";
        ret += instr.get_name();
        ret += "(const T a, [[maybe_unused]] const T b) noexcept\n";
        ret += "{\n    return ";
        ret += instr.get_code();
        ret += ";\n}\n\n";

        return ret;
    }

    template <typename iset>
    std::string generate_instructions_set()
    {
        std::string ret;
        const iset set{};
        std::apply([&ret](const auto &...i)
                   { ((ret += generate_function(i)), ...); },
                   set);
        return ret;
    }

    struct InstructionInfo
    {
        Instructions::InstructionID id{Instructions::InstructionID::nop};
        uint32_t op{0};
        std::string name{"invalid"};
        std::string code{""};
    };

    template <typename INSTR_SET>
    struct CodeMapping
    {

        std::vector<InstructionInfo> set;

        CodeMapping() noexcept
        {
            INSTR_SET is{};
            constexpr auto count = std::tuple_size<INSTR_SET>();
            set.resize(count);
            Get(is, set);
        }

        template <size_t I = 0, typename... Ts>
        constexpr void Get(std::tuple<Ts...> tup, std::vector<InstructionInfo> &dst)
        {
            if constexpr (I == sizeof...(Ts))
            {
                return;
            }
            else
            {
                const auto ii = std::get<I>(tup);
                dst[I].id = ii.id;
                dst[I].op = (uint32_t)ii.operands;
                dst[I].name = ii.get_name();
                dst[I].code = ii.get_code();

                Get<I + 1>(tup, dst);
            }
        }
    };
}