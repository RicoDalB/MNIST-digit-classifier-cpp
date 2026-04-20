#pragma once

#include <vector>

#include "../math/matrix.h"
#include "../nn/neural_network.h"

struct PredictionResult {
    int predicted_class;
    double confidence;
    std::vector<double> probabilities;
};

class Predictor {
public:
    Predictor();

    void load_model(const char* file_path);
    PredictionResult predict(const Matrix& input);

private:
    NeuralNetwork network_;
};