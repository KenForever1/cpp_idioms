// Wrapper.cpp
#include <memory>
#include "wrapper.h"
#include <iostream>

struct Wrapper::Wrapped {
    int a = 1;
};

static Wrapper::Wrapped* get_wrapped(Wrapper* wrapper) {
    // c++17 compatible
    return std::launder(reinterpret_cast<Wrapper::Wrapped*>(&wrapper->storage));
}

Wrapper::Wrapper() {
    std::cout << "sizeof Warpped: " << sizeof(Wrapped) << std::endl;
    std::cout << "sizeof storage: " << sizeof(this->storage) << std::endl;
    static_assert(sizeof(Wrapped) <= sizeof(this->storage) , "Object can't fit into local storage");
    this->handle = new (&this->storage) Wrapped();
}

Wrapper::~Wrapper() {
    handle->~Wrapped();
}

void Wrapper::print(){
    auto impl = get_wrapped(this);
    std::cout << "call print : "  << impl->a << std::endl;
}

