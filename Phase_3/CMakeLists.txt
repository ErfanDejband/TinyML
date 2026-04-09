# CMakeLists.txt — Your build recipe
cmake_minimum_required(VERSION 3.16)
project(TinyML_Main)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable from main.cpp
add_executable(main main.cpp)