#include "neuro.h"
#include <cmath>
#include <vector>

MLP::MLP(const vector<size_t> &layer_sizes) : layer_sizes(layer_sizes) {
    // Initialize the weights and biases for each layer
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        // Add the weight matrix for this layer
        weights.emplace_back(layer_sizes[i] * layer_sizes[i + 1]);

        // Add the bias vector for this layer
        biases.emplace_back(layer_sizes[i + 1]);

        weighted_inputs.emplace_back(layer_sizes[i + 1]);

        // Initialize the weights and biases with random values
        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0);
        for (double &j: weights[i]) {
            j = distribution(generator);
        }
        for (double &j: biases[i]) {
            j = distribution(generator);
        }
    }
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

double MLP::calculate_error(vector<double> output, const vector<double> &targets, vector<double> &output_error) {
    for (size_t j = 0; j < output.size(); j++) {
        output_error[j] = output[j] - targets[j];
    }

    // Calculate the mean squared error for the output layer
    double mean_squared_error = 0.0;
    for (double j: output_error) {
        mean_squared_error += pow(j, 2);
    }
    return mean_squared_error / output_error.size();
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

//        // Update the weights and biases for the input layer
//        for (size_t j = 0; j < layer_sizes[0]; j++) {
//            for (size_t k = 0; k < layer_sizes[1]; k++) {
//                weights[0][k * layer_sizes[0] + j] -=
//                        learning_rate * output_error[k] * sigmoid_derivative(output[k]) * inputs[j];
//            }
//            biases[0][j] -= learning_rate * output_error[j] * sigmoid_derivative(output[j]);
//        }
}

void MLP::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, size_t epochs,
                double learning_rate) {
    // Make sure the number of input rows matches the number of target rows
    if (inputs.size() != targets.size()) {
        throw runtime_error("Number of input rows does not match number of target rows");
    }
    for (size_t epoch = 0; epoch < epochs; epoch++) {
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
            double mean_squared_error = calculate_error(output, targets[indices[i]], output_error);
//                cout << i << ". MSE = " << mean_squared_error << endl;
            epoch_loss += mean_squared_error;
            apply_backpropagation(inputs[indices[i]], output, output_error, learning_rate);
        }
        // Calculate and print the average loss for this epoch
        epoch_loss /= inputs.size();
        cout << "Epoch " << epoch + 1 << " loss: " << epoch_loss << endl;
    }
}