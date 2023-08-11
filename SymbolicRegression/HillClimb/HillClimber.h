#pragma once

#include "EvaluatedCode.h"

namespace SymbolicRegression::HillClimb
{
	template <typename T>
	class HillClimber
	{
	public:
		HillClimber() = default;
		HillClimber(const CodeSettings &cs) noexcept
			: mCurrent(cs), mBest(cs)
		{
		}

		auto &Current() noexcept
		{
			return mCurrent;
		}

		const auto &Best() const noexcept
		{
			return mBest;
		}

		auto &Best() noexcept
		{
			return mBest;
		}

	private:
		EvaluatedCode<T> mCurrent{};
		EvaluatedCode<T> mBest{};

	public:
		std::vector<size_t> mSmallSet;
		std::vector<size_t> mBigSet;
		std::pair<size_t, double> mWorstBatch;
	};
}
