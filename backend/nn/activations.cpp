#include "activations.h"    
#include <cmath>

namespace Activations{
    Matrix ReLu(const Matrix& input){
        Matrix output(input.rows(), input.cols());
        for(int i = 0; i < input.rows(); i++){
            for(int j = 0; j < input.cols(); j++){
                double value = input.at(i, j);
                output.at(i, j) = (value > 0.0) ? value : 0.0;
            }
        }
        return output;
    }

    Matrix Sigmoid(const Matrix& input){
        Matrix output(input.rows(), input.cols());
        for(int i = 0; i < input.rows(); i++){
            for(int j = 0; j < input.cols(); j++){
                double value = input.at(i, j);
                output.at(i, j) = 1.0 / (1.0 + std::exp(-value));
            }
        }
        return output;
    }

    Matrix relu_derivative(const Matrix& grad_output, const Matrix& input) {
        if (grad_output.rows() != input.rows() || grad_output.cols() != input.cols()) {
            throw std::invalid_argument("ReLU derivative requires matching shapes.");
        }

        Matrix grad_input(input.rows(), input.cols());

        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                grad_input.at(i, j) = (input.at(i, j) > 0.0) ? grad_output.at(i, j) : 0.0;
            }
        }

        return grad_input;
    }

    Matrix softmax(const Matrix& input) {
        if (input.rows() != 1) {
            throw std::invalid_argument("Softmax currently expects a single row vector.");
        }

        Matrix output(1, input.cols());

        // Numerical stability trick:
        // subtract max before exponentiating
        double max_value = input.at(0, 0);
        for (int j = 1; j < input.cols(); ++j) {
            if (input.at(0, j) > max_value) {
                max_value = input.at(0, j);
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < input.cols(); ++j) {
            double exp_value = std::exp(input.at(0, j) - max_value);
            output.at(0, j) = exp_value;
            sum_exp += exp_value;
        }

        for (int j = 0; j < input.cols(); ++j) {
            output.at(0, j) /= sum_exp;
        }

        return output;
    }
}