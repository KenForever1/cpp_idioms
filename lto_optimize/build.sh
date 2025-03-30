#!/bin/bash
# build.sh
set -e

# 清理旧文件
rm -rf math_lib/*.o *.a main_*

# # 编译LTO版本
# cd math_lib
# g++ -c -flto -O2 square.cpp -o square.o
# ar cr libmath.a square.o
# cd ..

# g++ -flto -O2 main.cpp -Imath_lib -Lmath_lib -lmath -o main_lto

# # 编译无LTO版本
# cd math_lib
# g++ -c -O2 square.cpp -o square_no_lto.o
# ar cr libmath_no_lto.a square_no_lto.o
# cd ..

# g++ -O2 main.cpp -Imath_lib -Lmath_lib -lmath_no_lto -o main_no_lto

# echo "对比可执行文件大小:"
# ls -lh main_*
