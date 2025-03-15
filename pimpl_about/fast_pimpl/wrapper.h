#include <type_traits>
#include <cstddef>

// Wrapper.hpp
struct Wrapper {
    Wrapper();
    ~Wrapper();

    void print();
    
    // deprecated in C++23
    std::aligned_storage_t<32, alignof(std::max_align_t)> storage;
    
    struct Wrapped; // forward declaration
    Wrapped* handle;
};