#pragma once
#include <vector>
#include "layer.h"
#include "../math/matrix.h"

class NeuralNetwork{
public:
    NeuralNetwork(int input_size, const std::vector<int>& hidden_sizes, int output_size);
    Matrix forward(const Matrix& input); // Run the full forward pass
    void backward(const Matrix& grad_loss);
    void update_parameters(double learning_rate);
    double train_sample(const Matrix& input, const Matrix& output, double learning_rate);
    void print_structure() const;
    std::vector<DenseLayer>& layers();
    const std::vector<DenseLayer>& layers() const; 

private:
    int input_size_;
    int output_size_;
    std::vector<int> hidden_sizes_;
    // store all in order 
    std::vector<DenseLayer> layers_;
};