#include <iostream>
#include <vector>
#include <algorithm>

void func()
{
    std::vector<int> v1;
    // 18只是举例子，25或者其它任何数都可以
    for (auto i = 0; i < 99; i++)
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