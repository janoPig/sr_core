#pragma once

namespace SymbolicRegression::Computer::CodeGen
{
    template <typename T>
    constexpr const char *type_name() noexcept
    {
        if constexpr (std::is_same<T, bool>::value)
            return "bool";
        if constexpr (std::is_same<T, int32_t>::value)
            return "int32_t";
        if constexpr (std::is_same<T, uint32_t>::value)
            return "uint32_t";
        if constexpr (std::is_same<T, float>::value)
            return "float";
        if constexpr (std::is_same<T, double>::value)
            return "double";
        return "unknow_type";
    }

    template <typename T, typename instruction>
    std::string generate_function(const instruction &instr)
    {
        auto type = type_name<T>();
        std::string ret;
        ret += "inline ";
        ret += instr.get_name();
        ret += "(const ";
        ret += type;
        ret += " a, const ";
        ret += type;
        ret += " b) noexcept\n";
        ret += "{\n    return ";
        ret += instr.get_code();
        ret += ";\n}\n";

        return ret;
    }

    template <typename T, typename iset>
    std::string generate_instructions_set(const iset &set)
    {
        std::string ret;
        std::apply([&ret](const auto &...i)
                   { ((ret += generate_function<T>(i)), ...); },
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