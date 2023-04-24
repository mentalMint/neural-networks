#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

class MLP {
private:
    vector<size_t> m_layer_sizes;
    vector<vector<double>> m_weights;
    vector<vector<double>> m_biases;
    vector<vector<double>> activations;

public:
    explicit MLP(const vector<size_t> &layer_sizes) : m_layer_sizes(layer_sizes) {
        // Initialize the weights and biases for each layer
        for (size_t i = 0; i < m_layer_sizes.size() - 1; i++) {
            // Add the weight matrix for this layer
            m_weights.emplace_back(m_layer_sizes[i] * m_layer_sizes[i + 1]);

            // Add the bias vector for this layer
            m_biases.emplace_back(m_layer_sizes[i + 1]);

            // Initialize the weights and biases with random values
            default_random_engine generator;
            normal_distribution<double> distribution(0.0, 1.0);
            for (double & j : m_weights[i]) {
                j = distribution(generator);
            }
            for (double & j : m_biases[i]) {
                j = distribution(generator);
            }
        }
    }

    vector<double> feedforward(const vector<double> &input) {
        // Make sure the input size matches the first layer size
        if (input.size() != m_layer_sizes[0]) {
            throw runtime_error("Input size does not match first layer size");
        }

        // Initialize the activations for the input layer
        activations = {input};

        // Calculate the activations for each layer
        for (size_t i = 0; i < m_layer_sizes.size() - 1; i++) {
            // Calculate the weighted inputs for this layer
            vector<double> weighted_inputs(m_layer_sizes[i + 1]);
            for (size_t j = 0; j < m_layer_sizes[i + 1]; j++) {
                double weighted_input = 0.0;
                for (size_t k = 0; k < m_layer_sizes[i]; k++) {
                    weighted_input += m_weights[i][j * m_layer_sizes[i] + k] * activations[i][k];
                }
                weighted_inputs[j] = weighted_input + m_biases[i][j];
            }

            // Apply the activation function to the weighted inputs
            vector<double> layer_activations(m_layer_sizes[i + 1]);
            for (size_t j = 0; j < m_layer_sizes[i + 1]; j++) {
                layer_activations[j] = sigmoid(weighted_inputs[j]);
            }

            // Add the layer activations to the list of activations
            activations.push_back(layer_activations);
        }

        // Return the final layer activations
        return activations.back();
    }

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, size_t epochs,
               double learning_rate) {
        // Make sure the number of input rows matches the number of target rows
        if (inputs.size() != targets.size()) {
            throw runtime_error("Number of input rows does not match number of target rows");
        }

        // Train the MLP for the specified number of epochs
        for (size_t epoch = 0; epoch < epochs; epoch++) {
            // Shuffle the input and target rows
            vector<size_t> indices(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                indices[i] = i;
            }
            shuffle(indices.begin(), indices.end(), default_random_engine());

            // Train the MLP on each input and target row
            for (size_t i = 0; i < inputs.size(); i++) {
                // Feed the input forward through the MLP
                vector<double> output = feedforward(inputs[indices[i]]);
                // Calculate the error for the output layer
                vector<double> output_error(output.size());
                for (size_t j = 0; j < output.size(); j++) {
                    output_error[j] = output[j] - targets[indices[i]][j];
                    cout << output_error[j] << " ";
                }

                // Backpropagate the error through the MLP
                for (size_t j = m_layer_sizes.size() - 1; j > 0; j--) {
                    // Calculate the error for this layer
                    vector<double> layer_error(m_layer_sizes[j]);
                    for (size_t k = 0; k < m_layer_sizes[j]; k++) {
                        double weighted_error = 0.0;
                        for (size_t l = 0; l < m_layer_sizes[j - 1]; l++) {
                            weighted_error += m_weights[j - 1][k * m_layer_sizes[j - 1] + l] * layer_error[l];
                        }
                        layer_error[k] = sigmoid_derivative(output[k]) * weighted_error;
                    }

                    // Update the weights and biases for this layer
                    for (size_t k = 0; k < m_layer_sizes[j]; k++) {
                        for (size_t l = 0; l < m_layer_sizes[j - 1]; l++) {
                            m_weights[j - 1][k * m_layer_sizes[j - 1] + l] -=
                                    learning_rate * layer_error[k] * activations[j - 1][l];
                        }
                        m_biases[j - 1][k] -= learning_rate * layer_error[k];
                    }
                }

                // Update the weights and biases for the input layer
                for (size_t j = 0; j < m_layer_sizes[0]; j++) {
                    for (size_t k = 0; k < m_layer_sizes[1]; k++) {
                        m_weights[0][k * m_layer_sizes[0] + j] -=
                                learning_rate * output_error[k] * inputs[indices[i]][j];
                    }
                    m_biases[0][j] -= learning_rate * output_error[j];
                }
            }
        }
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }
};

#endif //PERCEPTRON_NEURO_H
