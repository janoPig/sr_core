#!/bin/bash

clear
echo buid...

clang++-15 *.cpp -o ./bin/hroch.bin -DNDEBUG -fveclib=SVML -std=c++20 -O3 -mavx2 -Wall -Wextra -funsafe-math-optimizations -ftree-vectorize -fno-exceptions -shared -fPIC #-fsanitize=address

echo done.
