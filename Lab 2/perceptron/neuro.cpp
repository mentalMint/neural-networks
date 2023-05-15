#include "neuro.h"
#include <cmath>
#include <vector>
#include <fstream>

#define SIGMOID_PARAM 2.5

MLP::MLP(const vector<size_t> &layer_sizes) : layer_sizes(layer_sizes) {
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        weights.emplace_back(layer_sizes[i] * layer_sizes[i + 1]);
        biases.emplace_back(layer_sizes[i + 1]);
        weighted_inputs.emplace_back(layer_sizes[i + 1]);

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0);
        for (double &j: weights[i]) {
            j = distribution(generator);
        }
        for (double &j: biases[i]) {
            j = distribution(generator);
        }
    }
//    mse_file << "Epoch, Value" << endl;
//    mae_file << "Epoch, Value" << endl;
//    r_squared_file << "Epoch, Value" << endl;
    results << "Epoch";
    for (int i = 0; i < layer_sizes[layer_sizes.size() - 1]; i++) {
        results << ",Predicted " << i << ",Real " << i;
    }
    results << endl;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-SIGMOID_PARAM * x));
}

double sigmoid_derivative(double x) {
    return SIGMOID_PARAM * sigmoid(x) * (1.0 - sigmoid(x));
}

vector<double> MLP::feedforward(const vector<double> &input) {
    // Make sure the input size matches the first layer size
    if (input.size() != layer_sizes[0]) {
        throw runtime_error("Input size does not match first layer size");
    }

    // Initialize the activations for the input layer
    activations = {input};

    // Calculate the activations for each layer
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        // Calculate the weighted inputs for this layer
//            vector<double> weighted_inputs(layer_sizes[i + 1]);
        for (size_t j = 0; j < layer_sizes[i + 1]; j++) {
            double weighted_input = 0.0;
            for (size_t k = 0; k < layer_sizes[i]; k++) {
                weighted_input += weights[i][j * layer_sizes[i] + k] * activations[i][k];
            }
            weighted_inputs[i][j] = weighted_input + biases[i][j];
        }

        // Apply the activation function to the weighted inputs
        vector<double> layer_activations(layer_sizes[i + 1]);
        for (size_t j = 0; j < layer_sizes[i + 1]; j++) {
            layer_activations[j] = sigmoid(weighted_inputs[i][j]);
        }

        // Add the layer activations to the list of activations
        activations.push_back(layer_activations);
    }

    // Return the final layer activations
    return activations.back();
}

void MLP::calculate_error(vector<double> output, const vector<double> &targets, vector<double> &output_error) {
    for (size_t j = 0; j < output.size(); j++) {
        output_error[j] = output[j] - targets[j];
        results << "," << output[j] << "," << targets[j];
    }
    results << endl;

//    double mean_squared_error_part = 0.0;
//    for (double j: output_error) {
//        mean_squared_error_part += pow(j, 2);
//    }
//    mean_squared_error_part /= output_error.size();
//    mean_squared_error += mean_squared_error_part;
//
//    double mean_absolute_error_part = 0.0;
//    for (double j: output_error) {
//        mean_absolute_error_part += abs(j);
//    }
//    mean_absolute_error_part /= output_error.size();
//    mean_absolute_error += mean_absolute_error_part;
}

void MLP::calculate_gradients(size_t layer_number, vector<double> &layer_error, vector<double> &perv_layer_error,
                              vector<double> output_error) {
    if (layer_number == layer_sizes.size() - 1) {
        for (size_t j = 0; j < layer_sizes[layer_number]; j++) {
            layer_error[j] = output_error[j] * sigmoid_derivative(weighted_inputs[layer_number - 1][j]);
        }
    } else {
        for (size_t k = 0; k < layer_sizes[layer_number]; k++) {
            double weighted_error = 0.0;
            for (size_t l = 0; l < layer_sizes[layer_number - 1]; l++) {
                weighted_error +=
                        weights[layer_number - 1][k * layer_sizes[layer_number - 1] + l] * perv_layer_error[l];
            }
            layer_error[k] = sigmoid_derivative(weighted_inputs[layer_number - 1][k]) * weighted_error;
        }
    }
}

void MLP::apply_backpropagation(const vector<double> &inputs, const vector<double> &output,
                                const vector<double> &output_error,
                                double learning_rate) {
    vector<double> prev_layer_error;
    for (size_t j = layer_sizes.size() - 1; j > 0; j--) {
        vector<double> layer_error(layer_sizes[j]);
        calculate_gradients(j, layer_error, prev_layer_error, output_error);
        prev_layer_error = layer_error;
        // Update the weights and biases for this layer
        for (size_t k = 0; k < layer_sizes[j]; k++) {
            for (size_t l = 0; l < layer_sizes[j - 1]; l++) {
                weights[j - 1][k * layer_sizes[j - 1] + l] -=
                        learning_rate * layer_error[k] * activations[j - 1][l];
            }
            biases[j - 1][k] -= learning_rate * layer_error[k];
        }
    }
}

void MLP::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, size_t epochs,
                double learning_rate) {
    if (inputs.size() != targets.size()) {
        throw runtime_error("Number of input rows does not match number of target rows");
    }
    for (size_t epoch = 0; epoch < epochs; epoch++) {
//        mean_squared_error = 0.0;
//        mean_absolute_error = 0.0;

        // Shuffle the input and target rows
        vector<size_t> indices(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            indices[i] = i;
        }
        shuffle(indices.begin(), indices.end(), default_random_engine());
        double epoch_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            vector<double> output = feedforward(inputs[indices[i]]);
            vector<double> output_error(output.size());
            results << epoch + 1;
            calculate_error(output, targets[indices[i]], output_error);
            apply_backpropagation(inputs[indices[i]], output, output_error, learning_rate);
        }

//        mean_squared_error /= inputs.size();
//        mean_absolute_error /= inputs.size();
//        mse_file << epoch + 1 << ", " << mean_squared_error << endl;
//        mae_file << epoch + 1 << ", " << mean_absolute_error << endl;
    }
}