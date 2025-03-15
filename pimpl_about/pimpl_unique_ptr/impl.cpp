#include "public.h"
#include <string>
#include <iostream>

// 定义 Impl 结构体（包含原私有成员）
struct MyClass::Impl {
    int data;
    std::string name;
    // 其他私有成员和辅助函数
};

// 构造函数：创建 Impl 实例
MyClass::MyClass() : pimpl(std::make_unique<Impl>()) {}

// 析构函数：需在 Impl 定义后实现
MyClass::~MyClass() = default;

// 方法通过 pimpl 访问实现
void MyClass::doSomething() {
    pimpl->data = 42;
    // 其他操作
    std::cout << "call doSomething" << std::endl;
}
