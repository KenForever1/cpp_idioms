#include <memory>

class MyClass {
public:
    MyClass();
    ~MyClass(); // 必须显式声明（析构需看到 Impl 定义）
    void doSomething();

    // 禁用拷贝（如需支持需手动实现深拷贝）
    MyClass(const MyClass&) = delete;
    MyClass& operator=(const MyClass&) = delete;

private:
    struct Impl;          // 前置声明
    std::unique_ptr<Impl> pimpl; // 指向实现的指针
};
