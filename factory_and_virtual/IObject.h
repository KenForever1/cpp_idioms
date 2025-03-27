#pragma once
#include <memory>

class IObject {
public:
    virtual ~IObject() = default;
    virtual void doWork() = 0;  // 纯虚接口
};

// 抽象工厂接口
class ObjectFactory {
public:
    virtual ~ObjectFactory() = default;
    virtual std::unique_ptr<IObject> create() = 0;
};

// 获取工厂实例（单例模式）
ObjectFactory& getObjectFactory();
