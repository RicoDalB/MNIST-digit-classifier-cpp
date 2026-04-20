#include "loss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Loss {

double mean_squared_error(const Matrix& prediction, const Matrix& target) {
    if (prediction.rows() != target.rows() || prediction.cols() != target.cols()) {
        throw std::invalid_argument("Prediction and target must have the same shape.");
    }

    double sum = 0.0;
    const int total_elements = prediction.rows() * prediction.cols();

    for (int row = 0; row < prediction.rows(); ++row) {
        for (int col = 0; col < prediction.cols(); ++col) {
            const double diff = prediction.at(row, col) - target.at(row, col);
            sum += diff * diff;
        }
    }

    return sum / static_cast<double>(total_elements);
}

Matrix mean_squared_error_derivative(const Matrix& prediction, const Matrix& target) {
    if (prediction.rows() != target.rows() || prediction.cols() != target.cols()) {
        throw std::invalid_argument("Prediction and target must have the same shape.");
    }

    const int total_elements = prediction.rows() * prediction.cols();
    Matrix gradient(prediction.rows(), prediction.cols());

    for (int row = 0; row < prediction.rows(); ++row) {
        for (int col = 0; col < prediction.cols(); ++col) {
            const double diff = prediction.at(row, col) - target.at(row, col);
            gradient.at(row, col) = 2.0 * diff / static_cast<double>(total_elements);
        }
    }

    return gradient;
}

double cross_entropy(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.rows() != target.rows() || probabilities.cols() != target.cols()) {
        throw std::invalid_argument("Probabilities and target dimensions must match for cross-entropy.");
    }

    const double epsilon = 1e-12;
    double loss = 0.0;

    for (int row = 0; row < probabilities.rows(); ++row) {
        for (int col = 0; col < probabilities.cols(); ++col) {
            const double clipped_probability =
                std::max(epsilon, std::min(1.0 - epsilon, probabilities.at(row, col)));

            loss += -target.at(row, col) * std::log(clipped_probability);
        }
    }

    return loss / static_cast<double>(probabilities.rows());
}

Matrix softmax_cross_entropy_gradient(const Matrix& probabilities, const Matrix& target) {
    if (probabilities.rows() != target.rows() || probabilities.cols() != target.cols()) {
        throw std::invalid_argument("Probabilities and target dimensions must match for gradient.");
    }

    Matrix gradient(probabilities.rows(), probabilities.cols());

    for (int row = 0; row < probabilities.rows(); ++row) {
        for (int col = 0; col < probabilities.cols(); ++col) {
            gradient.at(row, col) = probabilities.at(row, col) - target.at(row, col);
        }
    }

    return gradient;
}

}