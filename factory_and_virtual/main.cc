#include "IObject.h"

int main() {
    // 通过工厂创建对象（无需知道 ObjectImpl 的存在）
    auto obj = getObjectFactory().create();
    obj->doWork();  // 通过抽象接口调用
    return 0;
}
