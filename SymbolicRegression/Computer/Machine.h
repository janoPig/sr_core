#pragma once

#include "Processor.h"
#include "Memory.h"
#include "Code.h"
#include "../Utils/Dataset.h"
#include "../Utils/Evaluate.h"

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

		auto ComputeScore(
			const Dataset &data,
			const Code<T> &code,
			const std::vector<size_t> &batchSelection,
			Utils::Result &r,
			uint32_t transformation,
			uint32_t metric,
			T clipMin,
			T clipMax,
			T cw0,
			T cw1,
			bool filter = true,
			const Utils::BatchVector<T, BATCH> *sampleWeight = nullptr) noexcept
		{
			auto maxError = 0.0;
			size_t worstIdx = 0;
			T *__restrict yPred = mMemory[code.Size() - 1];
			const auto clip = clipMin < clipMax;
			const auto cw = cw0 != cw1;
			const T *__restrict sw = nullptr;
			if (sampleWeight)
				sw = sampleWeight->GetData();

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
				// TODO: refactor and move to a separate class
				auto score = 0.0;
				if (metric == 0)
				{
					if (sw)
						score = Utils::ComputeSqErr<T, true>(data.BatchY(batchIdx), yPred, BATCH, sw);
					else
						score = Utils::ComputeSqErr<T, false>(data.BatchY(batchIdx), yPred, BATCH);
				}
				else if (metric == 1)
				{
					if (sw)
						score = Utils::ComputeMAE<T, true>(data.BatchY(batchIdx), yPred, BATCH, sw);
					else
						score = Utils::ComputeMAE<T, false>(data.BatchY(batchIdx), yPred, BATCH);
				}
				else if (metric == 2)
				{
					if (sw)
						score = Utils::ComputeMSLE<T, true>(data.BatchY(batchIdx), yPred, BATCH, sw);
					else
						score = Utils::ComputeMSLE<T, false>(data.BatchY(batchIdx), yPred, BATCH);
				}
				else if (metric == 3)
				{
					score = 1.0 - std::abs(Utils::ComputePseudoKendall(data.BatchY(batchIdx), yPred, BATCH));
				}
				else if (metric == 4)
				{
					if (cw)
					{
						if (sw)
							score = Utils::ComputeLogLoss<T, true, true>(data.BatchY(batchIdx), yPred, BATCH, cw0, cw1, sw);
						else
							score = Utils::ComputeLogLoss<T, true, false>(data.BatchY(batchIdx), yPred, BATCH, cw0, cw1);
					}
					else
					{
						if (sw)
							score = Utils::ComputeLogLoss<T, false, true>(data.BatchY(batchIdx), yPred, BATCH, cw0, cw1, sw);
						else
							score = Utils::ComputeLogLoss<T, false, false>(data.BatchY(batchIdx), yPred, BATCH);
					}
				}

				r.mScore += score;

				if (score > maxError)
				{
					maxError = score;
					worstIdx = batchIdx;
				}
			}
			return std::pair{worstIdx, maxError / BATCH};
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