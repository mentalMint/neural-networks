#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


using namespace std;

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class MLP {
private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    double learning_rate;

public:
    MLP(const std::vector<int> &layer_sizes, double learning_rate)
            : layer_sizes(layer_sizes), learning_rate(learning_rate) {
        weights.resize(layer_sizes.size() - 1);
        biases.resize(layer_sizes.size() - 1);

        // Initialize weights and biases randomly
        srand(time(NULL));
        for (int i = 0; i < layer_sizes.size() - 1; i++) {
            weights[i].resize(layer_sizes[i + 1]);
            for (int j = 0; j < layer_sizes[i + 1]; j++) {
                weights[i][j].resize(layer_sizes[i]);
                for (int k = 0; k < layer_sizes[i]; k++) {
                    weights[i][j][k] = (double) rand() / RAND_MAX - 0.5;
                }
            }
            biases[i].resize(layer_sizes[i + 1]);
            for (int j = 0; j < layer_sizes[i + 1]; j++) {
                biases[i][j] = (double) rand() / RAND_MAX - 0.5;
            }
        }
    }

    std::vector<double> forward(const std::vector<double> &input) {
        std::vector<double> output = input;
        for (int i = 0; i < layer_sizes.size() - 1; i++) {
            std::vector<double> layer_output(layer_sizes[i + 1], 0.0);
            for (int j = 0; j < layer_sizes[i + 1]; j++) {
                for (int k = 0; k < layer_sizes[i]; k++) {
                    layer_output[j] += weights[i][j][k] * output[k];
                }
                layer_output[j] += biases[i][j];
                layer_output[j] = 1.0 / (1.0 + exp(-layer_output[j]));
            }
            output = layer_output;
        }
        return output;
    }

    void
    train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets, int epochs) {
        for (int i = 0; i < epochs; i++) {
            // Iterate over all input/target pairs
            for (int j = 0; j < inputs.size(); j++) {
                std::vector<double> input = inputs[j];
                std::vector<double> target = targets[j];

                // Forward pass
                std::vector<double> output = forward(input);

                // Calculate error
                std::vector<double> error(layer_sizes.back(), 0.0);
                for (int k = 0; k < layer_sizes.back(); k++) {
                    error[k] = output[k] - target[k];
                }

                // Backward pass
                std::vector<std::vector<double>> hidden_outputs(layer_sizes.size() - 2);
                hidden_outputs[0].resize(layer_sizes[1], 0.0);
                for (int k = 0; k < layer_sizes[1]; k++) {
                    double layer_output = 0.0;
                    for (int l = 0; l < layer_sizes[0]; l++) {
                        layer_output +=
                                weights[0][k][l] * input[l];
                    }
                    layer_output += biases[0][k];
                    hidden_outputs[0][k] = 1.0 / (1.0 + exp(-layer_output));
                }
                for (int k = 1; k < layer_sizes.size() - 2; k++) {
                    hidden_outputs[k].resize(layer_sizes[k + 1], 0.0);
                    for (int l = 0; l < layer_sizes[k + 1]; l++) {
                        double layer_output = 0.0;
                        for (int m = 0; m < layer_sizes[k]; m++) {
                            layer_output += weights[k][l][m] * hidden_outputs[k - 1][m];
                        }
                        layer_output += biases[k][l];
                        hidden_outputs[k][l] = 1.0 / (1.0 + exp(-layer_output));
                    }
                }

                std::vector<std::vector<double>> gradients(layer_sizes.size() - 1);
                for (int k = layer_sizes.size() - 2; k >= 0; k--) {
                    gradients[k].resize(layer_sizes[k + 1], 0.0);
                    if (k == layer_sizes.size() - 2) {
                        for (int l = 0; l < layer_sizes[k + 1]; l++) {
                            gradients[k][l] = error[l] * hidden_outputs[k - 1][l] * (1 - hidden_outputs[k - 1][l]);
                        }
                    } else {
                        for (int l = 0; l < layer_sizes[k + 1]; l++) {
                            double output_sum = 0.0;
                            for (int m = 0; m < layer_sizes[k + 2]; m++) {
                                output_sum += gradients[k + 1][m] * weights[k + 1][m][l];
                            }
                            gradients[k][l] = output_sum * hidden_outputs[k][l] * (1 - hidden_outputs[k][l]);
                        }
                    }
                }

                for (int k = 0; k < layer_sizes.size() - 1; k++) {
                    for (int l = 0; l < layer_sizes[k + 1]; l++) {
                        for (int m = 0; m < layer_sizes[k]; m++) {
                            weights[k][l][m] -= learning_rate * gradients[k][l] * output[m];
                        }
                        biases[k][l] -= learning_rate * gradients[k][l];
                    }
                }
            }
        }
    }

};


#endif //PERCEPTRON_NEURO_H
