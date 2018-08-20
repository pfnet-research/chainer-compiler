cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-trigraphs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

link_directories(onnx/.setuptools-cmake-build)
link_directories(googletest/googletest)

add_subdirectory(compiler)
add_subdirectory(common)
add_subdirectory(feeder)
add_subdirectory(runtime)
add_subdirectory(tools)
