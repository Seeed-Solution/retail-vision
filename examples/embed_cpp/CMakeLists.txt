# vb_algo embed example (spec BASE-1 §5.5.3 c, §6.13.2, M1.24).
cmake_minimum_required(VERSION 3.16)
project(embed_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(vb CONFIG REQUIRED)

add_executable(embed_demo main.cpp)
target_link_libraries(embed_demo PRIVATE vb::algo)
