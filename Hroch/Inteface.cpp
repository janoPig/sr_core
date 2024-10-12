#include "Inteface.h"
#include "SolverWrapper.h"
#include <thread>
#include <cstring>

using namespace Hroch;

struct SolverHandle
{
	std::vector<ISolver *> mSolvers;
	solver_params mSolverParams{};
};

template <typename T>
void FillDataset(Utils::Dataset<T, BATCH> &data, SymbolicRegression::Utils::BatchVector<T, BATCH> *sampleWeight, const T *X, const T *y, const T *sw, unsigned int rows, unsigned int xcols, uint64_t random_state) noexcept
{
	auto newSize = (rows / BATCH) * BATCH;
	if (newSize < rows)
		newSize += BATCH;
	for (unsigned int i = 0; i < xcols; i++)
	{
		std::memcpy(data.DataX(i), &X[i * rows], (size_t)rows * sizeof(T));
	}
	std::memcpy(data.DataY(), y, (size_t)rows * sizeof(T));
	if (sw && sampleWeight)
	{
		std::memcpy(sampleWeight->GetData(), sw, (size_t)rows * sizeof(T));
	}

	// Padding with random rows
	if (newSize > rows)
	{
		SymbolicRegression::Utils::RandomEngine re{};
		re.Seed(random_state);
		for (auto j = rows; j < newSize; j++)
		{
			const auto sampleId = re.Rand(rows);
			for (unsigned int i = 0; i < xcols; i++)
			{
				data.DataX(i)[j] = data.DataX(i)[sampleId];
			}
			data.DataY()[j] = data.DataY()[sampleId];
			if (sw && sampleWeight)
			{
				sampleWeight->GetData()[j] = sw[sampleId];
			}
		}
	}
}

std::vector<std::string> split(const std::string &target, char c)
{
	std::string temp;
	std::stringstream stringstream{target};
	std::vector<std::string> result;

	while (std::getline(stringstream, temp, c))
	{
		result.push_back(temp);
	}

	return result;
}

auto GetInstructions(const std::string &mt, uint32_t verbose)
{
	if (mt == "fuzzy")
		return SymbolicRegression::Computer::Instructions::FuzzyMath;
	if (mt == "simple")
		return SymbolicRegression::Computer::Instructions::BasicMath;
	if (mt == "math")
		return SymbolicRegression::Computer::Instructions::AdvancedMath;

	const SymbolicRegression::Computer::CodeGen::CodeMapping<SymbolicRegression::Computer::Instructions::Set> cm;
	std::vector<std::pair<SymbolicRegression::Computer::Instructions::InstructionID, double>> instrSet{};

	auto vec = split(mt, ';');
	for (const auto &str : vec)
	{
		std::stringstream ss{str};
		std::string instr;
		double prob;
		ss >> instr >> prob;

		if (prob < 0.0)
		{
			if (verbose)
				printf("error: instruction probability < 0\n");
			prob = 0.0;
		}

		bool find = false;
		for (const auto &x : cm.set)
		{
			if (x.name == instr)
			{
				find = true;
				instrSet.push_back({x.id, prob});
				break;
			}
		}
		if (!find)
		{
			if (verbose)
				printf("error: unknown instruction %s\n", instr.c_str());
		}
	}

	return instrSet;
}

auto GetFeatProbs(std::string str, uint32_t count, uint32_t verbose)
{
	if (str == "xicor")
	{
		// lets empty, compute from xicor
		return std::vector<std::pair<uint32_t, double>>{};
	}
	std::vector<std::pair<uint32_t, double>> featProbs{count};
	for (uint32_t i = 0; i < featProbs.size(); i++)
	{
		featProbs[i].first = i;
		featProbs[i].second = 1.0;
	}
	auto vec = split(str, ';');
	uint32_t i = 0;
	for (const auto &s : vec)
	{
		std::stringstream ss{s};
		double prob;
		ss >> prob;

		if (prob < 0.0)
		{
			if (verbose)
				printf("error: feature probability < 0\n");
			prob = 0.0;
		}
		if (i > featProbs.size() - 1)
		{
			if (verbose)
				printf("error: feature probability index > features count\n");
			break;
		}
		featProbs[i].second = prob;
		i++;
	}

	return featProbs;
}

SymbolicRegression::Config GetConfig(const solver_params &sp) noexcept
{
	return SymbolicRegression::Config{
		sp.random_state,
		sp.pop_size,
		sp.transformation,
		sp.clip_min,
		sp.clip_max,
		{sp.init_const_min,
		 sp.init_const_max,
		 sp.init_predefined_const_prob,
		 (sp.init_predefined_const_count && sp.init_predefined_const_set) ? std::vector<double>{sp.init_predefined_const_set, sp.init_predefined_const_set + sp.init_predefined_const_count} : std::vector<double>{}},
		{sp.input_size,
		 sp.const_size,
		 sp.min_code_size,
		 sp.max_code_size}};
}

SymbolicRegression::FitParams GetFitParams(const fit_params &fp, uint32_t xcols)
{
	return SymbolicRegression::FitParams{
		fp.time_limit,
		fp.verbose,
		fp.pop_sel,
		fp.metric,
		fp.pretest_size,
		fp.sample_size,
		fp.neighbours_count,
		fp.alpha,
		fp.beta,
		fp.iter_limit,
		{fp.const_min,
		 fp.const_max,
		 fp.predefined_const_prob,
		 (fp.predefined_const_count && fp.predefined_const_set) ? std::vector<double>{fp.predefined_const_set, fp.predefined_const_set + fp.predefined_const_count} : std::vector<double>{}},
		GetInstructions(fp.problem, fp.verbose),
		GetFeatProbs(fp.feature_probs, xcols, 0),
		{fp.cw0, fp.cw1},
	};
}

template <typename T>
int FitData(SolverHandle &solver,
			const SymbolicRegression::Utils::Dataset<T, BATCH> &data,
			const SymbolicRegression::FitParams &fp,
			SymbolicRegression::Utils::BatchVector<T, BATCH> *sw)
{
	if (fp.mVerbose > 1)
		printf("run fit task...\n");
	auto thread_func = [&](size_t idx)
	{
		solver.mSolvers[idx]->Fit(data, fp, sw);
	};

	std::vector<std::thread> threads;
	for (size_t i = 0; i < solver.mSolvers.size(); i++)
	{
		threads.emplace_back(std::thread(thread_func, (size_t)i));
	}

	for (auto &th : threads)
	{
		if (th.joinable())
			th.join();
	}
	if (fp.mVerbose > 1)
		printf("%zu threads done..\n", threads.size());
	return 0;
}

template <typename T>
auto GetFeatProbsFromXicor(FitParams& fp, const SymbolicRegression::Utils::Dataset<T, BATCH>& data, uint32_t rows)
{
	fp.mFeatProbs.resize(data.CountX());
	for (uint32_t i = 0; i < fp.mFeatProbs.size(); i++)
	{
		fp.mFeatProbs[i].first = i;
		fp.mFeatProbs[i].second = Utils::Xicor(data.DataX(i), data.DataY(), rows);
		fp.mFeatProbs[i].second = std::max(fp.mFeatProbs[i].second, 0.0001);
	}
}

template <typename T>
int FitData(SolverHandle &solver, const T *X, const T *y, uint32_t rows, uint32_t xcols, const fit_params &params, const T *sw)
{
	if (!X || !y || xcols < 1 || xcols != solver.mSolverParams.input_size || rows < 4)
	{
		if (params.verbose > 0)
		{
			printf("error: Invalid parameters");
		}
		return 1;
	}

	auto fp = GetFitParams(params, xcols);
	const auto cs = SymbolicRegression::CodeSettings{solver.mSolverParams.input_size, solver.mSolverParams.const_size, solver.mSolverParams.min_code_size, solver.mSolverParams.max_code_size};
	SymbolicRegression::Utils::Dataset<T, BATCH> data{(size_t)rows, cs};

	SymbolicRegression::Utils::BatchVector<T, BATCH> sampleWeight{(size_t)rows};
	FillDataset(data, &sampleWeight, X, y, sw, rows, xcols, solver.mSolverParams.random_state);

	if (fp.mFeatProbs.empty())
	{
		GetFeatProbsFromXicor(fp, data, rows);
	}

	return FitData(solver, data, fp, sw ? &sampleWeight : nullptr);
}

template <typename T>
int Predict(SolverHandle &solver, const T *X, T *y, unsigned int rows, unsigned int xcols, [[maybe_unused]] const predict_params *params)
{
	ISolver *ps = nullptr;

	if (params->id != (uint32_t)-1)
	{
		ps = solver.mSolvers[params->id / solver.mSolverParams.pop_size];
	}
	else
	{
		auto bestScore = std::numeric_limits<double>::max();
		for (auto s : solver.mSolvers)
		{
			if (s->Score() < bestScore)
			{
				ps = s;
				bestScore = s->Score();
			}
		}
	}

	if (!ps)
		return 1;

	const auto cs = SymbolicRegression::CodeSettings{solver.mSolverParams.input_size, solver.mSolverParams.const_size, solver.mSolverParams.min_code_size, solver.mSolverParams.max_code_size};
	SymbolicRegression::Utils::Dataset<T, BATCH> data{(size_t)rows, cs};
	FillDataset(data, static_cast<SymbolicRegression::Utils::BatchVector<T, BATCH> *>(nullptr), X, y, static_cast<const T *>(nullptr), rows, xcols, solver.mSolverParams.random_state);

	if (params->id != (uint32_t)-1)
		ps->Predict(data, solver.mSolverParams.transformation, params->id % solver.mSolverParams.pop_size, (T)solver.mSolverParams.clip_min, (T)solver.mSolverParams.clip_max);
	else
		ps->Predict(data, solver.mSolverParams.transformation, (T)solver.mSolverParams.clip_min, (T)solver.mSolverParams.clip_max);

	memcpy(y, data.DataY(), rows * sizeof(T));
	return 0;
}

void *CreateSolver(const solver_params *params)
{
	const auto handle = new SolverHandle{};
	handle->mSolverParams = *params;
	auto cfg = GetConfig(*params);
	handle->mSolvers.resize(params->num_threads);
	SymbolicRegression::Utils::RandomEngine re;
	re.Seed(params->random_state);
	for (auto &s : handle->mSolvers)
	{
		cfg.mRandomSeed = re.RandU64();
		s = SolverFactory::Create(cfg, params->precision == 1 ? EDataType::F32 : EDataType::F64);
	}
	return (void *)handle;
}

void DeleteSolver(void *hsolver)
{
	delete (SolverHandle *)hsolver;
}

int FitData32(void *hsolver, const float *X, const float *y, unsigned int rows, unsigned int xcols, const fit_params *params, const float *sw, unsigned int sw_len)
{
	SolverHandle *solver = (SolverHandle *)hsolver;
	if (solver->mSolverParams.precision != 1)
		return 1;

	return FitData(*solver, X, y, rows, xcols, *params, sw_len == rows ? sw : nullptr);
}

int FitData64(void *hsolver, const double *X, const double *y, unsigned int rows, unsigned int xcols, const fit_params *params, const double *sw, unsigned int sw_len)
{
	SolverHandle *solver = (SolverHandle *)hsolver;
	if (solver->mSolverParams.precision != 2)
		return 1;

	return FitData(*solver, X, y, rows, xcols, *params, sw_len == rows ? sw : nullptr);
}

int Predict32(void *hsolver, const float *X, float *y, unsigned int rows, unsigned int xcols, [[maybe_unused]] const predict_params *params)
{
	auto solver = (SolverHandle *)hsolver;

	if (solver->mSolverParams.precision != 1)
		return 1;

	return Predict(*solver, X, y, rows, xcols, params);
}

int Predict64(void *hsolver, const double *X, double *y, unsigned int rows, unsigned int xcols, [[maybe_unused]] const predict_params *params)
{
	auto solver = (SolverHandle *)hsolver;

	if (solver->mSolverParams.precision != 2)
		return 1;

	return Predict(*solver, X, y, rows, xcols, params);
}

void GetModel(const SymbolicRegression::HillClimb::CodeInfo info, math_model *model)
{
	model->score = info.mScore;
	model->partial_score = info.mPartialScore;
	model->str_representation = new char[info.mEquation.size() + 1];
	model->str_code_representation = new char[info.mCode.size() + 1];
#ifdef _WIN32
	strcpy_s(model->str_representation, info.mEquation.size() + 1, info.mEquation.c_str());
	strcpy_s(model->str_code_representation, info.mCode.size() + 1, info.mCode.c_str());
#else
	strcpy(model->str_representation, info.mEquation.c_str());
	strcpy(model->str_code_representation, info.mCode.c_str());
#endif
	model->used_constants_count = static_cast<uint32_t>(info.mConstants.size());
	if (model->used_constants_count)
	{
		model->used_constants = new double[info.mConstants.size()];
		for (size_t i = 0; i < info.mConstants.size(); i++)
		{
			model->used_constants[i] = info.mConstants[i];
		}
	}
}

int GetBestModel(void *hsolver, math_model *model)
{
	SolverHandle &solver = *((SolverHandle *)hsolver);
	auto bestScore = std::numeric_limits<double>::max();
	ISolver *best = nullptr;
	for (auto s : solver.mSolvers)
	{
		if (s->Score() < bestScore)
		{
			best = s;
			bestScore = s->Score();
		}
	}
	if (best)
	{
		const auto info = best->GetBestInfo();
		GetModel(info, model);
		return 0;
	}
	return 1;
}

int GetModel(void *hsolver, unsigned long long id, math_model *model)
{
	SolverHandle &solver = *((SolverHandle *)hsolver);
	const auto thread_id = id / solver.mSolverParams.pop_size;
	const auto pop_id = id % solver.mSolverParams.pop_size;
	if (thread_id > solver.mSolvers.size())
		return 1;

	const auto info = solver.mSolvers[thread_id]->GetInfo(thread_id, pop_id);

	GetModel(info, model);
	model->id = id;

	return 0;
}

void FreeModel(math_model *model)
{
	delete[] model->str_representation;
	delete[] model->str_code_representation;
	delete[] model->used_constants;
}

double Xicor32(const float *X, const float *y, unsigned int rows)
{
	return Utils::Xicor(X, y, rows);
}

double Xicor64(const double *X, const double *y, unsigned int rows)
{
	return Utils::Xicor(X, y, rows);
}

double Pearson32(const float *X, const float *y, unsigned int rows)
{
	return Utils::Pearson(X, y, rows);
}

double Pearson64(const double *X, const double *y, unsigned int rows)
{
	return Utils::Pearson(X, y, rows);
}
