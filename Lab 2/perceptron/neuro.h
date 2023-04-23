#include <iostream>
#include <cmath>
#include <vector>


#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


using namespace std;

class Perceptron {
private:
    int input_size;
    vector<double> weights;
    double bias;
    double learning_rate;

    double (* activation_function)(double);

    double (* activation_function_derivative)(double);

public:
    Perceptron(int input_size, double learning_rate, double (* activation_function)(double),
               double (* activation_function_derivative)(double));

    double feed_forward(vector<double> inputs);

    void train(vector<double> inputs, double target);
};

#endif //PERCEPTRON_NEURO_H
