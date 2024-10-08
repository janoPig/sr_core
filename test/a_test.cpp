#include <fstream>
#include <iostream>
#include <chrono>
using namespace std::chrono;

#include "../SymbolicRegression/SymbolicRegression.h"
#include "../Csv/CsvFile.h"

namespace Srl = SymbolicRegression;
using DataType = float;      // float or  double
constexpr size_t BATCH = 64; // for float must be multiple of 8, for double must be multiple of 4

using Dataset = Srl::Utils::Dataset<DataType, BATCH>;

using SRSolver = Srl::HillClimb::Solver<DataType, BATCH, true>;

auto testStart = high_resolution_clock::now();
inline void test_callback(const uint64_t it, const double err)
{
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - testStart);
    std::cout << "[" << it << "] "
              << " time: " << duration.count() * 0.001 << "   MSE: " << err << "   RMSE: " << sqrt(err) << std::endl;
}

void run_test(const char *path, size_t iter_count = (size_t)-1)
{
    CsvFile csv(path);
    const auto samplesCount = csv.RowsCount();

    Srl::Config cfg{
        .mRandomSeed = 42,
        .mPopulationSize = 64,
        .mTransformation = 0,
        .mClipMin = 0.0,
        .mClipMax = 0.0,
        .mInitConstSettings = {.mMin = -1.0, .mMax = 1.0, .mPredefinedProb = 0.001, .mPredefinedSet = {0.0, 1.0, -1.0, 3.141592654}},
        .mCodeSettings = {csv.ColumnsCount() - 1, 8, 32, 32}};

    Dataset data{samplesCount, cfg.mCodeSettings};

    auto newSize = (samplesCount / BATCH) * BATCH;
    if (newSize < samplesCount)
        newSize += BATCH;

    Srl::Utils::RandomEngine re{};
    re.Seed(42);
    for (size_t i = 0; i < newSize; i++)
    {
        auto sampleId = i;
        if (sampleId >= samplesCount)
        {
            // Padding with random rows
            sampleId = re.Rand(samplesCount);
        }
        const auto &row = csv[sampleId];
        for (size_t x = 0; x < row.size() - 1; x++)
        {
            data.SetX(x, i, static_cast<DataType>(row[x]));
        }
        data.SetY(i, static_cast<DataType>(row.back()));
    }

    cfg.mCodeSettings.mInputSize = (uint32_t)data.CountX();

    std::cout << path << " loaded..." << std::endl;

    std::vector<std::pair<uint32_t, double>> featProbs(csv.ColumnsCount() - 1);
    for (uint32_t i = 0; i < featProbs.size(); i++)
    {
        featProbs[i].first = i;
        featProbs[i].second = Srl::Utils::Xicor(data.DataX(i), data.DataY(), samplesCount);
        featProbs[i].second = std::max(featProbs[i].second, 0.0001);
    }

    Srl::FitParams fp{
        .mTimeLimit = 0,
        .mVerbose = 2,
        .mTournament = 4,
        .mMetric = 0, // MSE
        .mPretestSize = 1,
        .mSampleSize = 16,
        .mNeighboursCount = 15,
        .mAlpha = 0.15,
        .mBeta = 0.5,
        .mIterLimit = iter_count,
        .mConstSettings = {.mMin = -1e30, .mMax = 1e30, .mPredefinedProb = 0.001, .mPredefinedSet = {0.0, 1.0, -1.0, 3.141592654}},
        .mInstrProbs = Srl::Computer::Instructions::AdvancedMath,
        .mFeatProbs = featProbs };

    SRSolver solver{cfg};

    testStart = high_resolution_clock::now();
    solver.Fit(data, fp, test_callback, nullptr);

    // const auto info = solver.GetBestInfo();
    // std::cout << info.mCode;
    // const auto info2 = solver.GetInfo(0, 0);
    // std::cout << info2.mCode;
}

int main(int /*argc*/, char * /*argv*/[])
{
    // run_test("../test/344_mv.tsv" /*, 10000*/);
    run_test("../test/586_fri_c3_1000_25.tsd" /*, 10000*/);

    return 0;
}
