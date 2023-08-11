#pragma once

namespace SymbolicRegression::Computer
{

    template <typename T, size_t BATCH>
    class Memory
    {
        Memory() = delete;
        Memory(const Memory &) = delete;
        Memory(Memory &&) = delete;
        Memory &operator=(const Memory &) = delete;
        Memory &operator=(Memory &&) = delete;

    public:
        explicit Memory(const CodeSettings &cs) noexcept
            : mCodeSegment((cs.mMaxCodeSize) * BATCH),
              mMem(cs.MaxMemorySize(), nullptr)
        {
            for (size_t i = 0; i < cs.mMaxCodeSize; i++)
            {
                mMem[i] = mCodeSegment.GetBatch(i);
            }
        }

        ~Memory() = default;

        const auto &operator[](const size_t n) const noexcept
        {
            return mMem[n];
        }

        auto &operator[](const size_t n) noexcept
        {
            return mMem[n];
        }

    private:
        Utils::BatchVector<T, BATCH> mCodeSegment;
        std::vector<T *> mMem;
    };
}
