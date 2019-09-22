cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mnist)

find_package(Torch REQUIRED)

add_library(mnist SHARED mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})

add_executable(test test.cpp mnist.cpp)
target_compile_features(test PUBLIC cxx_range_for)
target_link_libraries(test ${TORCH_LIBRARIES})
