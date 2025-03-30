#!/bin/bash

echo "查看LTO版本符号:"
# 查看LTO版本符号
nm --demangle main_lto | grep square
# 理想结果：无square符号（函数被内联）

echo "查看no_LTO版本符号:"

# 查看无LTO版本符号
nm --demangle main_no_lto | grep square
# 预期输出：存在square符号


echo "生成LTO汇编:"
# 生成带LTO的汇编
g++ -flto -O2 -S main.cpp -Imath_lib -Lmath_lib -lmath -o main_lto.s

echo "生成no_LTO汇编:"
# 生成无LTO的汇编
g++ -O2 -S main.cpp -Imath_lib -Lmath_lib -lmath_no_lto -o main_no_lto.s

