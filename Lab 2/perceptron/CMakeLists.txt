cmake_minimum_required(VERSION 3.24)
project(perceptron)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-O3")
add_executable(perceptron neuro.cpp neuro.h csv.cpp csv.h mushrooms_main.cpp MLPClassifier.cpp MLPClassifier.h)
