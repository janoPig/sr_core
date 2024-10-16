#pragma once
#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

struct solver_params
{
    unsigned long long random_state;
    unsigned int num_threads;
    unsigned int precision; // float32=1, float64=2
    unsigned int pop_size;
    unsigned int transformation;
    double clip_min;
    double clip_max;
    unsigned int input_size;
    unsigned int const_size;
    unsigned int min_code_size;
    unsigned int max_code_size;
    double init_const_min;
    double init_const_max;
    double init_predefined_const_prob;
    unsigned int init_predefined_const_count;
    const double *init_predefined_const_set;
};

struct fit_params
{
    unsigned int time_limit;
    unsigned int verbose;
    unsigned int pop_sel;
    unsigned int metric;
    unsigned int pretest_size; // batch count
    unsigned int sample_size;  // batch count
    unsigned int neighbours_count;
    double alpha;
    double beta;
    unsigned long long iter_limit;
    double const_min;
    double const_max;
    double predefined_const_prob;
    unsigned int predefined_const_count;
    const double *predefined_const_set;
    const char *problem;
    const char *feature_probs;
    double cw0;
    double cw1;
};

struct predict_params
{
    unsigned long long id;
    unsigned int verbose;
};

struct math_model
{
    unsigned long long id;
    double score;
    double partial_score;
    char *str_representation;
    char *str_code_representation;
    unsigned long used_constants_count;
    double *used_constants;
};

extern "C" EXPORT void *CreateSolver(const solver_params *params);
extern "C" EXPORT void DeleteSolver(void *hsolver);
extern "C" EXPORT int FitData32(void *hsolver, const float *X, const float *y, unsigned int rows, unsigned int xcols, const fit_params *params, const float *sw, unsigned int sw_len);
extern "C" EXPORT int FitData64(void *hsolver, const double *X, const double *y, unsigned int rows, unsigned int xcols, const fit_params *params, const double *sw, unsigned int sw_len);
extern "C" EXPORT int Predict32(void *hsolver, const float *X, float *y, unsigned int rows, unsigned int xcols, const predict_params *params);
extern "C" EXPORT int Predict64(void *hsolver, const double *X, double *y, unsigned int rows, unsigned int xcols, const predict_params *params);
extern "C" EXPORT int GetBestModel(void *hsolver, math_model *model);
extern "C" EXPORT int GetModel(void *hsolver, unsigned long long id, math_model *model);
extern "C" EXPORT void FreeModel(math_model *model);
extern "C" EXPORT double Xicor32(const float *X, const float *y, unsigned int rows);
extern "C" EXPORT double Xicor64(const double *X, const double *y, unsigned int rows);
extern "C" EXPORT double Pearson32(const float *X, const float *y, unsigned int rows);
extern "C" EXPORT double Pearson64(const double *X, const double *y, unsigned int rows);
