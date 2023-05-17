//
// Created by rshuv on 17.05.2023.
//

#ifndef MAIN_CPP_MLPCLASSIFIER_H
#define MAIN_CPP_MLPCLASSIFIER_H


#include "neuro.h"

class MLPClassifier : public MLP {
private:
    void calculate_layer_error(size_t layer_number, vector<double> &layer_error, vector<double> &prev_layer_error,
                               vector<double> output_error) override;

public:
    explicit MLPClassifier(const vector<size_t> &layerSizes);

    MLPClassifier(const vector<size_t> &layerSizes, const string &logFileName);
};


#endif //MAIN_CPP_MLPCLASSIFIER_H
