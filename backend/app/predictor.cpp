#include "predictor.h"

#include <stdexcept>

#include "../io/weights_io.h"
#include "../nn/activations.h"

namespace {
int argmax(const Matrix& row_vector) {
    if (row_vector.rows() != 1) {
        throw std::invalid_argument("argmax expects a 1 x N row vector.");
    }

    int best_index = 0;
    double best_value = row_vector.at(0, 0);

    for (int col = 1; col < row_vector.cols(); ++col) {
        if (row_vector.at(0, col) > best_value) {
            best_value = row_vector.at(0, col);
            best_index = col;
        }
    }

    return best_index;
}
}

Predictor::Predictor()
    : network_(784, {64, 32}, 10) {
}

void Predictor::load_model(const char* file_path) {
    WeightsIO::load_network(network_, file_path);
}

PredictionResult Predictor::predict(const Matrix& input) {
    Matrix logits = network_.forward(input);
    Matrix probabilities_matrix = Activations::softmax(logits);

    int predicted = argmax(probabilities_matrix);
    double confidence = probabilities_matrix.at(0, predicted);

    std::vector<double> probabilities;
    probabilities.reserve(10);

    for (int i = 0; i < 10; ++i) {
        probabilities.push_back(probabilities_matrix.at(0, i));
    }

    return PredictionResult{predicted, confidence, probabilities};
}