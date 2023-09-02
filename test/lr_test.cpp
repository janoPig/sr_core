#include <fstream>
#include <iostream>
#include <chrono>
using namespace std::chrono;

#include "../SymbolicRegression/SymbolicRegression.h"
#include "../Csv/CsvFile.h"
#include "../SymbolicRegression/Utils/LinearRegression.h"

using namespace SymbolicRegression::Utils;
#if 0
int main(int /*argc*/, char * /*argv*/[])
{
    double X1[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double X2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const double *X[2] = {nullptr, X2};
    double y[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double B[2] = {};
    linear_regression_2<double>(X, y, B, 10);
    return 0;
}
#endif