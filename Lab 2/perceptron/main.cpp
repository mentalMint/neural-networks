#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "my_neuro.h"
#include "csv_reader.h"
#include "neuro.h"

using namespace std;

//double sigmoid(double x) {
//    return 1 / (1 + exp(-x));
//}
//
//double sigmoid_derivative(double sigmoid) {
//    return sigmoid * (1 - sigmoid);
//}


vector<vector<string>> parse_csv(const string &filename, char delimiter = ',') {
    // Open the CSV file for reading
    ifstream file(filename);

    // Initialize the result vector
    vector<vector<string>> result;

    // Read each line of the file and split it into fields
    string line;
    while (getline(file, line)) {
        vector<string> fields;

        // Use a stringstream to split the line into fields
        stringstream ss(line);
        string field;
        while (getline(ss, field, delimiter)) {
            fields.push_back(field);
        }

        // Add the fields to the result vector
        result.push_back(fields);
    }

    // Close the file and return the result vector
    file.close();
    return result;
}


vector<vector<string>> remove_column(const vector<vector<string>> &data, int col_index) {
    // Create a new vector to hold the result
    vector<vector<string>> result;

    // Iterate over the rows of the input data
    for (const auto &i: data) {
        // Create a new row vector to hold the filtered columns
        vector<string> row;

        // Iterate over the columns of the input data
        for (size_t j = 0; j < i.size(); ++j) {
            // Skip the column to be removed
            if (j == col_index) {
                continue;
            }
            row.push_back(i[j]);
        }

        // Add the filtered row to the result vector
        result.push_back(row);
    }

    return result;
}

vector<vector<string>> remove_row(const vector<vector<string>> &data, int row_index) {
    // Create a new vector to hold the result
    vector<vector<string>> result;

    // Iterate over the rows of the input data
    for (size_t i = 0; i < data.size(); ++i) {
        // Skip the row to be removed
        if (i == row_index) {
            continue;
        }

        // Add the row to the result vector
        result.push_back(data[i]);
    }

    return result;
}

vector<string> get_column(const vector<vector<string>> &data, int col_index) {
    // Create a new vector to hold the result
    vector<string> result;

    // Iterate over the rows of the input data
    for (size_t i = 0; i < data.size(); ++i) {
        // Add the value in the desired column to the result vector
        result.push_back(data[i][col_index]);
    }

    return result;
}

vector<vector<double>> convert_to_doubles(const vector<vector<string>> &data) {
    // Create a new vector to hold the result
    vector<vector<double>> result;

    // Iterate over the rows of the input data
    for (size_t i = 0; i < data.size(); ++i) {
        // Create a new row vector to hold the converted values
        vector<double> row;

        // Iterate over the columns of the input data
        for (size_t j = 0; j < data[i].size(); ++j) {
            // Convert the string value to a double
            double value;
            istringstream(data[i][j]) >> value;

            // Add the converted value to the row vector
            row.push_back(value);
        }

        // Add the converted row to the result vector
        result.push_back(row);
    }

    return result;
}

vector<double> convert_to_doubles(const vector<string> &data) {
    // Create a new vector to hold the result
    vector<double> result;

    // Iterate over the input data
    for (const auto &value: data) {
        // Convert the string value to a double
        double double_value;
        istringstream(value) >> double_value;

        // Add the converted value to the result vector
        result.push_back(double_value);
    }

    return result;
}

size_t get_num_columns(const vector<vector<double>> &data) {
    // If there are no rows, return 0
    if (data.empty()) {
        return 0;
    }

    // Return the number of columns in the first row
    return data[0].size();
}

void print_table(const vector<vector<string>> &data) {
    for (const auto &row: data) {
        for (const auto &field: row) {
            cout << field << " ";
        }
        cout << endl;
    }
}

vector<vector<string>> get_last_two_columns(const vector<vector<string>> &table) {
    vector<vector<string>> result(table.size());
    for (size_t i = 0; i < table.size(); i++) {
        result[i].resize(2);
        result[i][0] = table[i][table[i].size() - 2];
        result[i][1] = table[i][table[i].size() - 1];
    }
    return result;
}

double* vectorOfVectorsToArray(vector<vector<double>> v) {
    int rows = v.size();
    int cols = v[0].size();

    double* arr = new double[rows * cols];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i * cols + j] = v[i][j];
        }
    }

    return arr;
}

vector<double> get_row(vector<vector<double>> matrix, int row) {
    return matrix[row];
}


int main() {
    vector<std::vector<std::string>> dataset = parse_csv("//mnt/d/Studies/Neuro/Lab 2/perceptron/prepared_dataset.csv");
    vector<std::vector<std::string>> dataset_without_target_string = remove_column(dataset, 0);
//    print_table(dataset_without_target_string);
//    vector<std::string> g_total_string = get_column(dataset_without_target_string, 23);
//    vector<std::string> kgf_string = get_column(dataset_without_target_string, 23);
    dataset_without_target_string = remove_row(dataset_without_target_string, 0);
    vector<std::vector<std::string>> targets_string = get_last_two_columns(dataset_without_target_string);
    dataset_without_target_string = remove_column(dataset_without_target_string, 23);
    dataset_without_target_string = remove_column(dataset_without_target_string, 23);
    vector<std::vector<double>> dataset_without_target = convert_to_doubles(dataset_without_target_string);
//    vector<double> g_total = convert_to_doubles(g_total_string);
//    vector<double> kgf = convert_to_doubles(kgf_string);
    vector<std::vector<double>> targets = convert_to_doubles(targets_string);

    size_t features_num = get_num_columns(dataset_without_target);

//    MLP mlp({get_num_columns(dataset_without_target), 50, 50, 2});
//    mlp.train(dataset_without_target, targets, 1, 0.001);
//
//    vector<double> result = mlp.feedforward(dataset_without_target[0]);
//    cout << endl;
//    cout << "Predicted: " << result[0] << " " << result[1] << endl;
//    cout << "Real: " << targets[0][0] << " " << targets[0][1] << endl;


    std::vector<std::vector<double>> inputs = {{0, 0},
                                               {0, 1},
                                               {1, 0},
                                               {1, 1}};
    std::vector<std::vector<double>> targets2 = {{0},
                                                {1},
                                                {1},
                                                {0}};
    MLP mlp({2, 3, 1}, 0.5);
    mlp.train(inputs, targets2, 1000);

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> input = inputs[i];
        std::vector<double> output = mlp.forward(input);
        std::cout << "Input: " << input[0] << " " << input[1] << " Output: " << output[0] << std::endl;
    }



//    auto sizes = new uint16_t[]{23, 50, 50, 2};
//    double* data = vectorOfVectorsToArray(dataset_without_target);
//    double* dtargets = vectorOfVectorsToArray(targets);
//    NeuralNet neuralNet(4, sizes);
//    neuralNet.learnBackpropagation(data, dtargets, 0.01, 100);
//    neuralNet.Forward(features_num, data);
//    double output[2] = {0};
//    neuralNet.getResult(2, output);
//    cout << "Predicted: " << output[0] << " " << output[1] << endl;
//    cout << "Real: " << targets[0][0] << " " << targets[0][1] << endl;
    return 0;
}