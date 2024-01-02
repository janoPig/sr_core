#pragma once
#include "Hash.h"

namespace SymbolicRegression::Utils
{
    class RandomEngine
    {
        uint64_t mState = 0xF1EC'5CA3'5791'DF4Bull;

    public:
        void Seed(uint64_t seed) noexcept
        {
            mState ^= seed;
        }

        inline auto State() const noexcept
        {
            return mState;
        }

        uint64_t RandU64() noexcept
        {
            mState = Mix64(mState);
            return mState;
        }

        bool Rand() noexcept
        {
            return RandU64() & 1;
        }

        bool TestProb(const uint64_t prob, const uint64_t X = 1023ULL) noexcept
        {
            return (RandU64() & X) < prob;
        }

        uint64_t Rand(const uint64_t range) noexcept
        {
            return RandU64() % range;
        }

        uint32_t Rand(const uint32_t range) noexcept
        {
            return (uint32_t)(RandU64() % range);
        }

        float Rand(const float range) noexcept
        {
            return range * static_cast<float>(RandU64()) / static_cast<float>(0xFFFFFFFFFFFFFFFFULL);
        }

        double Rand(const double range) noexcept
        {
            return range * static_cast<double>(RandU64()) / static_cast<double>(0xFFFFFFFFFFFFFFFFULL);
        }

        template <typename T>
        T Rand(const T minVal, const T maxVal) noexcept
        {
            return minVal + Rand(maxVal - minVal);
        }

        template <typename V>
        auto RandomElement(const V &vec) noexcept
        {
            return vec[Rand(vec.size())];
        }

        template <class RandomIt>
        void Shuffle(RandomIt first, RandomIt last)
        {
            typedef typename std::iterator_traits<RandomIt>::difference_type diff_t;

            for (diff_t i = last - first - 1; i > 0; --i)
            {
                using std::swap;
                std::swap(first[i], first[Rand((uint32_t)(i + 1))]);
            }
        }
    };

    template <typename T>
    class DiscreteRandomVariable
    {
    public:
        DiscreteRandomVariable(std::vector<std::pair<T, double>> probs)
            : mValues(probs.size())
        {
            double sum = 0.0;
            for (size_t i = 0; i < probs.size(); i++)
            {
                mValues[i] = probs[i].first;
                sum += probs[i].second;
            }
            for (auto &x : probs)
            {
                x.second = x.second / sum;
            }
            mAlias = GenerateAliasTable(probs);
        }

        T operator()(RandomEngine &re) const noexcept
        {
            const size_t idx = re.Rand(mAlias.size());
            if (re.Rand(0.0, 1.0) >= mAlias[idx].first and
                mAlias[idx].second != std::numeric_limits<size_t>::max())
            {
                return mValues[mAlias[idx].second];
            }
            else
            {
                return mValues[idx];
            }
        }

    private:
        std::vector<std::pair<double, size_t>> GenerateAliasTable(const std::vector<std::pair<T, double>> &probs)
        {
            const size_t sz = probs.size();
            std::vector<std::pair<double, size_t>> alias(sz, {0.0, std::numeric_limits<size_t>::max()});
            std::queue<size_t> small, large;

            for (size_t i = 0; i != sz; ++i)
            {
                alias[i].first = sz * probs[i].second;
                if (alias[i].first < 1.0)
                {
                    small.push(i);
                }
                else
                {
                    large.push(i);
                }
            }

            while (not(small.empty()) and not(large.empty()))
            {
                auto s = small.front(), l = large.front();
                small.pop(), large.pop();
                alias[s].second = l;
                alias[l].first -= (1.0 - alias[s].first);

                if (alias[l].first < 1.0)
                {
                    small.push(l);
                }
                else
                {
                    large.push(l);
                }
            }

            return alias;
        }

    private:
        std::vector<T> mValues;
        std::vector<std::pair<double, size_t>> mAlias;
    };
}
