#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

using namespace std;

class MLP {
private:
    vector<size_t> layer_sizes;
    vector<vector<double>> weights;
    vector<vector<double>> biases;
    vector<vector<double>> activations;
    vector<vector<double>> weighted_inputs;
//    double mean_squared_error = 0;
//    double mean_absolute_error = 0;
//    double r_squared = 0;
//    ofstream mse_file = ofstream("mse.csv", ofstream::trunc);
//    ofstream mae_file = ofstream("mae.csv", ofstream::trunc);
//    ofstream r_squared_file = ofstream("r_squared.csv", ofstream::trunc);
    ofstream results = ofstream("results.csv", ofstream::trunc);

    void apply_backpropagation(const vector<double> &inputs, const vector<double> &output,
                               const vector<double> &output_error,
                               double learning_rate);

    void calculate_error(vector<double> output, const vector<double> &targets, vector<double> &output_error);

    void calculate_gradients(size_t layer_number, vector<double> &layer_error, vector<double> &perv_layer_error,
                             vector<double> output_error);
public:
    explicit MLP(const vector<size_t> &layer_sizes);

    vector<double> feedforward(const vector<double> &input);

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, size_t epochs,
               double learning_rate);
};

#endif //PERCEPTRON_NEURO_H
