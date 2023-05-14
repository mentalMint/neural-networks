#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>


#define SIGMOID_PARAM 2.5

using namespace std;

class MLP {
private:
    vector<size_t> layer_sizes;
    vector<vector<double>> weights;
    vector<vector<double>> biases;
    vector<vector<double>> activations;
    vector<vector<double>> weighted_inputs;

    void apply_backpropagation(const vector<double> &inputs, const vector<double> &output,
                               const vector<double> &output_error,
                               double learning_rate);

    double calculate_error(vector<double> output, const vector<double> &targets, vector<double> &output_error);

    void calculate_gradients(size_t layer_number, vector<double> &layer_error, vector<double> &perv_layer_error,
                             vector<double> output_error);
public:
    explicit MLP(const vector<size_t> &layer_sizes);

    vector<double> feedforward(const vector<double> &input);

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, size_t epochs,
               double learning_rate);
};

#endif //PERCEPTRON_NEURO_H
