#!/bin/bash

export CXX=/usr/bin/clang++
export CC=/usr/bin/clang
cmake -B build -S .
cmake --build build