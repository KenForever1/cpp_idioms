#include <iostream>
#include <vector>
#include <algorithm>

void func()
{
    std::vector<int> v1;
    // 18只是举例子，25或者其它任何数都可以
    for (auto i = 0; i < 18; i++)
    {
        v1.push_back(121);
    }
    std::cout << "v1 addr : " << &v1[0] << std::endl;
    std::cout << "befor sort" << std::endl;
    std::sort(v1.begin(), v1.end(), [](int a, int b) { return a >= b; });
    std::cout << "after sort" << std::endl;
}

int main(int, char **)
{
    func();
}
// 报错gdb可以查看vector析构报错
// Program returned: 139
// double free or corruption (out)
// Program terminated with signal: SIGSEGV
// v1 addr : 0x166c5370
// befor sort
// after sort

// you can use gdb show this info
// g++ -g main.cpp
// gdb a.out
// when i < 18
// (gdb) x/40wd 0x55555556d360
// 0x55555556d360: 0       0       145     0
// 0x55555556d370: 121     121     121     121
// 0x55555556d380: 121     121     121     121
// 0x55555556d390: 121     121     121     121
// 0x55555556d3a0: 121     121     121     121
// 0x55555556d3b0: 121     121     0       0
// 0x55555556d3c0: 0       0       0       0
// 0x55555556d3d0: 0       0       0       0
// 0x55555556d3e0: 0       0       0       0
// 0x55555556d3f0: 0       0       1041    0
// (gdb) x/40wd 0x55555556d360
// 0x55555556d360: 0       0       145     121
// 0x55555556d370: 121     0       121     121
// 0x55555556d380: 121     121     121     121
// 0x55555556d390: 121     121     121     121
// 0x55555556d3a0: 121     121     121     121
// 0x55555556d3b0: 121     121     0       0
// 0x55555556d3c0: 0       0       0       0
// 0x55555556d3d0: 0       0       0       0
// 0x55555556d3e0: 0       0       0       0
// 0x55555556d3f0: 0       0       1041    0

// when i < 25
// (gdb) x/40wd 0x55555556d360
// 0x55555556d360: 0       0       145     0
// 0x55555556d370: 121     121     121     121
// 0x55555556d380: 121     121     121     121
// 0x55555556d390: 121     121     121     121
// 0x55555556d3a0: 121     121     121     121
// 0x55555556d3b0: 121     121     121     121
// 0x55555556d3c0: 121     121     121     121
// 0x55555556d3d0: 121     0       0       0
// 0x55555556d3e0: 0       0       0       0
// 0x55555556d3f0: 0       0       1041    0
// (gdb) x/40wd 0x55555556d360
// 0x55555556d360: 0       0       145     121
// 0x55555556d370: 121     121     121     121
// 0x55555556d380: 121     121     121     121
// 0x55555556d390: 0       121     121     121
// 0x55555556d3a0: 121     121     121     121
// 0x55555556d3b0: 121     121     121     121
// 0x55555556d3c0: 121     121     121     121
// 0x55555556d3d0: 121     0       0       0
// 0x55555556d3e0: 0       0       0       0
// 0x55555556d3f0: 0       0       1041    0

// when you use std::string
// #include <iostream>
// #include <string>
// #include <vector>
// #include <algorithm>

// void func()
// {
//     std::vector<std::string> v1;

//     for (auto i = 0; i < 18; i++)
//     {
//         v1.push_back("123");
//     }
//     std::cout << "v1 addr : " << &v1[0] << std::endl;
//     std::cout << "befor sort" << std::endl;
//     std::sort(v1.begin(), v1.end(), [](std::string a, std::string b) { return a >= b; });
//     std::cout << "after sort" << std::endl;
// }

// int main(int, char **)
// {
//     func();
// }
// 不管是18,还是25,或者其它数,都会遇到内存错误,比如：
// v1 addr : 0x55b6b5ab76e0
// befor sort
// fish: Job 1, './a.out' terminated by signal SIGSEGV (Address boundary error)