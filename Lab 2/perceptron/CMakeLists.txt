cmake_minimum_required(VERSION 3.25)
project(perceptron)

set(CMAKE_CXX_STANDARD 14)

add_executable(perceptron main.cpp neuro.cpp neuro.h)
