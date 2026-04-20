#pragma once

#include "../math/matrix.h"

namespace Activations{
    Matrix ReLu(const Matrix& input);
    Matrix Sigmoid(const Matrix& input);
    Matrix relu_derivative(const Matrix& grad_output, const Matrix& input);
    Matrix softmax(const Matrix& input);
}