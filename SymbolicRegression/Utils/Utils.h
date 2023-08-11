#pragma once

namespace SymbolicRegression::Utils
{
    template <typename T>
    bool IsFinite(const T val) noexcept
    {
        return val == val && std::isfinite(val);
    }

    template <typename T>
    inline std::string ToStringWithPrecision(const T val, const int n = 20)
    {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << val;
        return out.str();
    }

    template <typename T>
    T *AlignedAlloc(size_t alignment, size_t size) noexcept
    {
#ifdef __GNUC__
        return static_cast<T *>(std::aligned_alloc(alignment, size * sizeof(T)));
#else
        return static_cast<T *>(_aligned_malloc(size * sizeof(T), alignment));
#endif
    }

    template <typename T>
    void AlignedFree(T *ptr) noexcept
    {

#ifdef __GNUC__
        std::free(static_cast<void *>(ptr));
#else
        _aligned_free(ptr);
#endif
    }

    template <typename T1, typename T2>
    struct tuple_cat_helper;
    template <typename... T1, typename... T2>
    struct tuple_cat_helper<std::tuple<T1...>, std::tuple<T2...>>
    {
        using type = std::tuple<T1..., T2...>;
    };

    template <typename T1, typename T2>
    using tuple_cat_t = typename tuple_cat_helper<T1, T2>::type;

    template <typename T1, typename T2>
    using tuple_cat_t = typename tuple_cat_helper<T1, T2>::type;
}
