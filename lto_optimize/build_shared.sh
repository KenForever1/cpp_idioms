#!/bin/bash
# build.sh
set -e

# 清理旧文件
rm -rf math_lib/*.o *.a *.so main_*


# 编译LTO版本动态库
cd math_lib
# 编译时需添加 -fPIC（位置无关代码），且需在编译和链接阶段均启用 -flto
g++ -c -fPIC -flto -O2 square.cpp -o square.o
# 生成动态库时使用 -shared 和 -flto
g++ -shared -flto -O2 -o libmath_lto.so square.o
cd ..

# 链接动态库时需确保库路径正确（运行时可能需要设置 LD_LIBRARY_PATH 或 -rpath）
g++ -O2 main.cpp -Imath_lib -Lmath_lib -lmath_lto -o main_lto

# 编译无LTO版本动态库
cd math_lib
# 非LTO版本仍需 -fPIC，但无需 -flto
g++ -c -fPIC -O2 square.cpp -o square_no_lto.o
g++ -shared -O2 -o libmath_no_lto.so square_no_lto.o
cd ..

# 链接无LTO动态库
g++ -O2 main.cpp -Imath_lib -Lmath_lib -lmath_no_lto -o main_no_lto

echo "对比可执行文件大小:"
ls -lh main_*
echo "对比动态库大小:"
ls -lh math_lib/libmath*.so
