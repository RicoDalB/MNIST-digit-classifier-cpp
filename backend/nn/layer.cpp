#include "layer.h"
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>

DenseLayer::DenseLayer(int input_size, int output_size)
    :input_size_(input_size), output_size_(output_size), weights_(input_size, output_size), bias_(1, output_size),
        last_input_(1, input_size), last_output_(1, output_size), grad_weights_(input_size, output_size), grad_bias_(1, output_size){
        if(input_size <= 0 || output_size <= 0){
            throw std::invalid_argument("Layer size must be positive");
        }
    initialize_parameters();
}

// Easy initialization parameters with Xavier uniform
void DenseLayer::initialize_parameters() {
    double limit = std::sqrt(6.0 / (input_size_ + output_size_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < input_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            weights_.at(i, j) = dist(gen);
        }
    }

    bias_.fill(0.0);
    grad_weights_.fill(0.0);
    grad_bias_.fill(0.0);
}

// Forword pass --> z = xW+b
Matrix DenseLayer::forward(const Matrix& input){
    if(input.rows() != 1 || input.cols() != input_size_){
        throw std::invalid_argument("Input shape dose not match layer input size");
    }
    last_input_ = input;
    Matrix weighted_sum = Matrix::multiply(input, weights_);
    Matrix output = Matrix::add(weighted_sum, bias_);

    last_output_ = output;
    return output;
}

Matrix DenseLayer::backward(const Matrix& grad_output){
    if(grad_output.rows() != 1 || grad_output.cols() != output_size_){
        throw std::invalid_argument("Gradient shape does not match layer output size");
    }
    // dW = XT * dZ
    Matrix input_transposed = last_input_.transponse();
    grad_weights_ = Matrix::multiply(input_transposed, grad_output);

    grad_bias_ = grad_output;

    //dX = dZ * WT
    Matrix weight_transpose = weights_.transponse();
    Matrix grad_input = Matrix::multiply(grad_output, weight_transpose);

    return grad_input;
}

void DenseLayer::upgrade_parameters(double learning_rate){
    if(learning_rate <= 0.0){
        throw std::invalid_argument("Learning rate must be positive");
    }
    for (int i = 0; i < weights_.rows(); ++i) {
        for (int j = 0; j < weights_.cols(); ++j) {
            weights_.at(i, j) -= learning_rate * grad_weights_.at(i, j);
        }
    }

    for (int j = 0; j < bias_.cols(); ++j) {
        bias_.at(0, j) -= learning_rate * grad_bias_.at(0, j);
    }

}

//print wheight and bias
void DenseLayer::print_parameters() const{
    std::cout << "Weights:\n";
    weights_.print();

    std::cout << "Bias:\n";
    bias_.print();
}

int DenseLayer::input_size() const{
    return input_size_;
}

int DenseLayer::output_size() const{
    return output_size_;
}

const Matrix& DenseLayer::last_output() const {
    return last_output_;
}

const Matrix& DenseLayer::weights() const {
    return weights_;
}

const Matrix& DenseLayer::bias() const {
    return bias_;
}

void DenseLayer::set_weights(const Matrix& weights) {
    if (weights.rows() != input_size_ || weights.cols() != output_size_) {
        throw std::invalid_argument("Weight matrix dimensions do not match layer dimensions.");
    }

    weights_ = weights;
}

void DenseLayer::set_bias(const Matrix& bias) {
    if (bias.rows() != 1 || bias.cols() != output_size_) {
        throw std::invalid_argument("Bias matrix dimensions do not match layer dimensions.");
    }

    bias_ = bias;
}