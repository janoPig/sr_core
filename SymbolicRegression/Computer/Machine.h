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

		void ComputeScore(
			const Dataset &data,
			const Code<T> &code,
			const std::vector<size_t> &batchSelection,
			Utils::Result<BATCH> &r,
			uint32_t transformation,
			uint32_t metric,
			T clipMin,
			T clipMax,
			T cw0,
			T cw1,
			bool filter = true,
			const Utils::BatchVector<T, BATCH> *sampleWeight = nullptr) noexcept
		{
			T *__restrict yPred = mMemory[code.Size() - 1];
			const auto clip = clipMin < clipMax;
			const auto cw = cw0 != cw1;

			for (const auto batchIdx : batchSelection)
			{
				const T *__restrict yTrue = data.BatchY(batchIdx);
				const T *__restrict sw = sampleWeight ? sampleWeight->GetBatch(batchIdx) : nullptr;

				mProcessor.Execute(code, data, mMemory, batchIdx, filter);

				auto score = 0.0;

				// Logit approximation directly compute score log(1+exp(−f(x))) resp. log(1+exp(f(x)))
				if (metric == 20)
				{
					if (cw)
					{
						if (sw)
							score = Utils::ComputeLogitApprox<T, true, true>(yTrue, yPred, BATCH, cw0, cw1, sw);
						else
							score = Utils::ComputeLogitApprox<T, true, false>(yTrue, yPred, BATCH, cw0, cw1);
					}
					else
					{
						if (sw)
							score = Utils::ComputeLogitApprox<T, false, true>(yTrue, yPred, BATCH, cw0, cw1, sw);
						else
							score = Utils::ComputeLogitApprox<T, false, false>(yTrue, yPred, BATCH);
					}
				}
				else
				{
					if (transformation)
					{
						Utils::TransformData<T, BATCH>(yPred, transformation);
					}

					if (clip)
					{
						Utils::Clip<T, BATCH>(yPred, clipMin, clipMax);
					}

					// TODO: refactor and move to a separate class
					if (metric == 0)
					{
						if (sw)
							score = Utils::ComputeSqErr<T, true>(yTrue, yPred, BATCH, sw);
						else
							score = Utils::ComputeSqErr<T, false>(yTrue, yPred, BATCH);
					}
					else if (metric == 1)
					{
						if (sw)
							score = Utils::ComputeMAE<T, true>(yTrue, yPred, BATCH, sw);
						else
							score = Utils::ComputeMAE<T, false>(yTrue, yPred, BATCH);
					}
					else if (metric == 2)
					{
						if (sw)
							score = Utils::ComputeMSLE<T, true>(yTrue, yPred, BATCH, sw);
						else
							score = Utils::ComputeMSLE<T, false>(yTrue, yPred, BATCH);
					}
					else if (metric == 3)
					{
						score = 1.0 - std::abs(Utils::ComputePseudoKendall(yTrue, yPred, BATCH));
					}
					else if (metric == 4)
					{
						if (cw)
						{
							if (sw)
								score = Utils::ComputeLogLoss<T, true, true>(yTrue, yPred, BATCH, cw0, cw1, sw);
							else
								score = Utils::ComputeLogLoss<T, true, false>(yTrue, yPred, BATCH, cw0, cw1);
						}
						else
						{
							if (sw)
								score = Utils::ComputeLogLoss<T, false, true>(yTrue, yPred, BATCH, cw0, cw1, sw);
							else
								score = Utils::ComputeLogLoss<T, false, false>(yTrue, yPred, BATCH);
						}
					}
				}

				r.Add(batchIdx, score);
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