#pragma once

#include "../math/matrix.h"

namespace Loss {
    double mean_squared_error(const Matrix& prediction, const Matrix& target);
    Matrix mean_squared_error_derivative(const Matrix& prediction, const Matrix& target);

    double cross_entropy(const Matrix& probabilities, const Matrix& target);
    Matrix softmax_cross_entropy_gradient(const Matrix& probabilities, const Matrix& target);
}