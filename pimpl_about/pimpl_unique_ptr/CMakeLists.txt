
project(main)


add_library(impl SHARED impl.cpp)

add_executable(main main.cpp)
target_link_libraries(main impl)