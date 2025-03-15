
project(main)


add_library(wrapper SHARED wrapper.cpp)

add_executable(main main.cpp)
target_link_libraries(main wrapper)