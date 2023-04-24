cmake_minimum_required(VERSION 3.25)
project(perceptron)

set(CMAKE_CXX_STANDARD 14)

add_executable(perceptron main.cpp my_neuro.cpp my_neuro.h csv_reader.cpp csv_reader.h neuro.cpp neuro.h)
