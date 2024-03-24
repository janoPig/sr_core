#!/bin/bash

clear
echo buid...

clang++-18 *.cpp -o ./bin/hroch.bin -DNDEBUG -fveclib=libmvec -std=c++20 -O3 -mavx2 -Wall -Wextra -fno-math-errno -fno-signed-zeros -funsafe-math-optimizations -ftree-vectorize -fno-exceptions -shared -fPIC #-fsanitize=address

echo done.
