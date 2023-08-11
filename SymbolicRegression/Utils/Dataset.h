#pragma once

#include "BatchVector.h"

namespace SymbolicRegression::Utils
{
    template <typename T, size_t BATCH, size_t ALIGN = 32>
    class Dataset
    {
        using BVector = BatchVector<T, BATCH, ALIGN>;

    public:
        Dataset(size_t size, const CodeSettings &cs)
            : mSize(size),
              mBatchCount(BVector::BatchCount(size)),
              mX(cs.mInputSize),
              mY(nullptr)
        {
            for (size_t i = 0; i < cs.mInputSize; i++)
            {
                mX[i] = std::make_unique<BVector>(size);
            }
            mY = std::make_unique<BVector>(size);
        }

        Dataset() = delete;
        ~Dataset() = default;
        Dataset(const Dataset &) = delete;
        Dataset(Dataset &&) = delete;
        Dataset &operator=(const Dataset &) = delete;
        Dataset &operator=(Dataset &&) = delete;

        size_t AddColumn()
        {
            mX.push_back(std::make_unique<BVector>(mSize));
            return mX.size() - 1;
        }

        size_t Size() const noexcept
        {
            return mSize;
        }

        size_t BatchCount() const noexcept
        {
            return mBatchCount;
        }

        size_t CountX() const noexcept
        {
            return mX.size();
        }

        T *BatchX(const size_t x, const size_t batchIndex) noexcept
        {
            assert(x < mX.size());
            return mX[x]->GetBatch(batchIndex);
        }

        T *const BatchX(const size_t x, const size_t batchIndex) const noexcept
        {
            assert(x < mX.size());
            return mX[x]->GetBatch(batchIndex);
        }

        T *BatchY(const size_t batchIndex) noexcept
        {
            return mY->GetBatch(batchIndex);
        }

        T *const BatchY(const size_t batchIndex) const noexcept
        {
            return mY->GetBatch(batchIndex);
        }

        void SetX(const size_t x, const size_t idx, const T value) noexcept
        {
            assert(x < mX.size());
            mX[x]->SetAt(idx, value);
        }

        void SetY(const size_t idx, const T value) noexcept
        {
            mY->SetAt(idx, value);
        }

        T *DataX(const size_t x) const noexcept
        {
            assert(x < mX.size());
            return mX[x]->GetData();
        }

        T *DataY() const noexcept
        {
            return mY->GetData();
        }

    private:
        const size_t mSize{};
        const size_t mBatchCount{};
        std::vector<std::unique_ptr<BVector>> mX{};
        std::unique_ptr<BVector> mY{};
    };
}
