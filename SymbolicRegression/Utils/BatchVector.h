#pragma once

namespace SymbolicRegression::Utils
{
    template <typename T, size_t BATCH, size_t ALIGN = 32>
    struct BatchVector
    {
        explicit BatchVector(const size_t size) noexcept
            : mSize(size),
              mPtr(Utils::AlignedAlloc<T>(ALIGN, BatchCount(size) * BATCH * sizeof(T)))
        {
        }

        ~BatchVector()
        {
            Utils::AlignedFree(mPtr);
        }

        BatchVector() = delete;
        BatchVector(const BatchVector &) = delete;
        BatchVector(BatchVector &&) = delete;
        BatchVector &operator=(const BatchVector &) = delete;
        BatchVector &operator=(BatchVector &&) = delete;

        const T *GetBatch(const size_t idx) const noexcept
        {
            assert(idx < BatchCount(mSize));
            return &mPtr[idx * BATCH];
        }

        T *GetBatch(const size_t idx) noexcept
        {
            assert(idx < BatchCount(mSize));
            return &mPtr[idx * BATCH];
        }

        size_t Size() const noexcept
        {
            return mSize;
        }

        T *GetData() const noexcept
        {
            return mPtr;
        }

        void SetAt(const size_t idx, const T val) noexcept
        {
            mPtr[idx] = val;
        }

        constexpr static size_t BatchCount(const size_t size) noexcept
        {
            const auto cnt = size / BATCH;
            return (size % BATCH) ? cnt + 1 : cnt;
        }

    private:
        const size_t mSize{};
        T *mPtr{nullptr};

        static_assert((sizeof(T) * BATCH >= ALIGN));
        static_assert((sizeof(T) * BATCH & (ALIGN - 1)) == 0);
    };
}