cmake_minimum_required(VERSION 3.10)
project(a_demo)

set(CMAKE_CXX_STANDARD 20)
add_library(e STATIC libE.cpp)
target_compile_options(e PUBLIC -fPIC)
add_library(c SHARED libC.cpp)

target_link_libraries(c
e
)

add_library(d SHARED libD.cpp)
target_link_libraries(d
e
)


add_executable(main
    main.cpp
)
target_link_libraries(main
    c
    d
)
