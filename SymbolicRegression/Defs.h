#pragma once

inline constexpr double LARGE_FLOAT = 1.0e30;

//#define LOG_ENABLED

#ifdef LOG_ENABLED
#include "Utils/Log.h"
#ifndef __FILENAME__
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#endif
#define LOG(message,...) SymbolicRegression::Utils::log.print(__FILENAME__ , __LINE__, message, __VA_ARGS__)
#else
#define LOG(message,...)
#endif

