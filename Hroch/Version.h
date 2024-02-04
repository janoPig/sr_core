#pragma once
#include <iostream>

namespace Hroch
{
    inline constexpr uint32_t VERSION_MAJOR = 1;
    inline constexpr uint32_t VERSION_MINOR = 4;
    inline constexpr uint32_t VERSION_REVISION = 9;

    inline void PrintVersion()
    {
        printf("hroch core version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION);
    }
}