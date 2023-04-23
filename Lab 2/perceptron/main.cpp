#include <iostream>
#include <cmath>
#include <vector>
#include "neuro.h"

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double sigmoid) {
    return sigmoid * (1 - sigmoid);
}

int main() {
    Perceptron perceptron(2, 0.001, sigmoid, sigmoid_derivative);

    // train OR gate
    vector<vector<double>> training_data = {{0, 0},
                                            {0, 1},
                                            {1, 0},
                                            {1, 1}};
    vector<double> targets = {0, 1, 1, 1};
    for (int i = 0; i < 1000000; i++) {
        int index = rand() % 4;
        perceptron.train(training_data[index], targets[index]);
    }

    // test OR gate
    for (int i = 0; i < 4; i++) {
        double result = perceptron.feed_forward(training_data[i]);
        cout << training_data[i][0] << " OR " << training_data[i][1] << " = "
             << (result > 0.5) << " (" << result << ") " << endl;
    }

    return 0;
}