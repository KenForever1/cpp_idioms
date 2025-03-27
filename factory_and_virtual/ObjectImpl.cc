#include "IObject.h"
#include <iostream>

// 具体实现类（对客户端完全隐藏）
class ObjectImpl : public IObject {
    private:
    int a = 3;
    std::string b = "hello";
public:
    void doWork() override {
        std::cout << "Working with hidden implementation\n";
        std::cout << "a " << a << " b " << b ;
    }
};

// 具体工厂实现
class HiddenFactory : public ObjectFactory {
public:
    std::unique_ptr<IObject> create() override {
        return std::make_unique<ObjectImpl>();
    }
};

// 返回工厂实例（隐藏工厂实现）
ObjectFactory& getObjectFactory() {
    static HiddenFactory factory;
    return factory;
}
