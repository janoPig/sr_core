#pragma once

#include "Processor.h"
#include "Memory.h"
#include "Code.h"
#include "../Utils/Dataset.h"
#include "../Utils/Evaluate.h"
#include "../Utils/LinearRegression.h"

namespace SymbolicRegression::Computer
{

	template <typename T, size_t BATCH>
	class Machine
	{
		using Dataset = Utils::Dataset<T, BATCH>;
		Machine() = delete;

	public:
		explicit Machine(const CodeSettings &cs) noexcept
			: mCodeSettings(cs),
			  mMemory(cs),
			  mProcessor(cs)
		{
		}

		auto ComputeScore(const Dataset &data, const Code<T> &code, const std::vector<size_t> &batchSelection, Utils::Result &r, uint32_t transformation, uint32_t metric, T clipMin, T clipMax, bool filter = true) noexcept
		{
			auto maxError = 0.0;
			size_t worstIdx = 0;
			T *__restrict yPred = mMemory[code.Size() - 1];
			const auto clip = clipMin < clipMax;

			for (const auto batchIdx : batchSelection)
			{
				mProcessor.Execute(code, data, mMemory, batchIdx, filter);

				if (transformation)
				{
					Utils::TransformData<T, BATCH>(yPred, transformation);
				}

				if (clip)
				{
					Utils::Clip<T, BATCH>(yPred, clipMin, clipMax);
				}

				r.mSamplesCount += BATCH;
				auto score = 0.0;
				if (metric == 0)
					score = Utils::ComputeSqErr(data.BatchY(batchIdx), yPred, BATCH);
				if (metric == 1)
					score = Utils::ComputeMAE(data.BatchY(batchIdx), yPred, BATCH);
				else if (metric == 2)
					score = Utils::ComputeMSLE(data.BatchY(batchIdx), yPred, BATCH);
				else if (metric == 3)
					score = 1.0 - std::abs(Utils::ComputePseudoKendall(data.BatchY(batchIdx), yPred, BATCH));
				else if (metric == 4)
					score = Utils::ComputeLogLoss(data.BatchY(batchIdx), yPred, BATCH);

				r.mScore += score;

				if (score > maxError)
				{
					maxError = score;
					worstIdx = batchIdx;
				}
			}
			return std::pair{worstIdx, maxError / BATCH};
		}

		auto ComputeScoreOLS(const Dataset &data, const Code<T> &code, const std::vector<size_t> &batchSelection, T *__restrict buff_x, T *__restrict buff_y) noexcept
		{
			struct Result
			{
				double err;
				T B0;
				T B1;
			};

			T *__restrict yPred = mMemory[code.Size() - 1];

			for (size_t i = 0; i < batchSelection.size(); i++)
			{
				const auto batchIdx = batchSelection[i];
				mProcessor.Execute(code, data, mMemory, batchIdx, true);
				const auto y = data.BatchY(batchIdx);

				auto *xptr = &buff_x[i * BATCH];
				auto *yptr = &buff_y[i * BATCH];

				for (size_t n = 0; n < BATCH; n++)
				{
					xptr[n] = yPred[n];
					yptr[n] = y[n];
				}
			}

			const T *X[2] = {nullptr, buff_x};
			T B[2] = {0, 1};

			Utils::linear_regression_2<T>(X, buff_y, B, batchSelection.size() * BATCH);

			auto err = 0.0;
			for (size_t i = 0; i < batchSelection.size() * BATCH; i++)
			{
				const auto y2 = B[0] + B[1] * buff_x[i];
				err += (buff_y[i] - y2) * (buff_y[i] - y2);
			}
			if (Utils::IsFinite(err))
			{
				err /= (batchSelection.size() * BATCH);
			}
			else
			{
				err = LARGE_FLOAT;
			}
			return Result{err , B[0], B[1]};
		}

		void ComputeOLS(Dataset &data, const Code<T> &code, T B0, T B1) noexcept
		{
			T *__restrict yPred = mMemory[code.Size() - 1];

			for (size_t batchIdx = 0; batchIdx < data.BatchCount(); batchIdx++)
			{
				mProcessor.Execute(code, data, mMemory, batchIdx, true);

				T *const __restrict y2 = data.BatchY(batchIdx);
				for (size_t n = 0; n < BATCH; n++)
				{
					y2[n] = B0 + B1 * yPred[n];
				}
			}
		}

		void Compute(Dataset &data, const Code<T> &code, uint32_t transformation, T clipMin, T clipMax, bool filter = false) noexcept
		{
			T *__restrict yPred = mMemory[code.Size() - 1];
			const auto clip = clipMin < clipMax;

			for (size_t batchIdx = 0; batchIdx < data.BatchCount(); batchIdx++)
			{
				mProcessor.Execute(code, data, mMemory, batchIdx, filter);

				if (transformation)
				{
					Utils::TransformData<T, BATCH>(yPred, transformation);
				}

				if (clip)
				{
					Utils::Clip<T, BATCH>(yPred, clipMin, clipMax);
				}

				T *const __restrict y2 = data.BatchY(batchIdx);
				for (size_t n = 0; n < BATCH; n++)
				{
					y2[n] = yPred[n];
				}
			}
		}

	private:
		const CodeSettings mCodeSettings{};
		Memory<T, BATCH> mMemory{};
		Processor<T, BATCH> mProcessor{};
	};
}