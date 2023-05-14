cmake_minimum_required(VERSION 3.24)
project(perceptron)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-O3")
add_executable(perceptron main.cpp neuro.cpp neuro.h csv_reader.cpp csv_reader.h)
