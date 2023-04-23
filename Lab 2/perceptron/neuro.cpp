#include "neuro.h"
#include <cmath>
#include <vector>

Perceptron::Perceptron(int input_size, double learning_rate, double (* activation_function)(double),
                       double (* activation_function_derivative)(double)) {
    this->input_size = input_size;
    this->learning_rate = learning_rate;
    this->activation_function = activation_function;
    this->activation_function_derivative = activation_function_derivative;
    this->bias = 0;

    // initialize weights randomly
    for (int i = 0; i < input_size; i++) {
        weights.push_back(((double) rand() / RAND_MAX) - 0.5);
    }
}


double Perceptron::feed_forward(vector<double> inputs) {
    double weighted_sum = 0;
    for (int i = 0; i < input_size; i++) {
        weighted_sum += inputs[i] * weights[i];
    }
    double activation = activation_function(weighted_sum + bias);
    return activation;
}

void Perceptron::train(vector<double> inputs, double target) {
    double output = feed_forward(inputs);
    double error = target - output;
    double gradient =activation_function_derivative(output) * error;
    for (int i = 0; i < input_size; i++) {
        weights[i] += learning_rate * gradient * inputs[i];
    }
    bias += learning_rate * gradient;
}
