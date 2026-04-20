#pragma once

#include "../math/matrix.h"

// Dense layer representation, it perform: output = input * weight + bias

class DenseLayer{
public: 
    DenseLayer(int input_size, int output_size);
    Matrix forward(const Matrix& input); 
    Matrix backward(const Matrix& grad_output);
    void upgrade_parameters(double learning_rate);
    void print_parameters() const; 
    int input_size() const;
    int output_size() const;
    const Matrix& last_output() const;
    const Matrix& weights() const;
    const Matrix& bias() const;
    void set_weights(const Matrix& weights);
    void set_bias(const Matrix& bias);

private:
    int input_size_;
    int output_size_;
    Matrix weights_;
    Matrix bias_;
    // Chached val for the most reacent forward pass
    Matrix last_input_;
    Matrix last_output_;
    // Gradients computd during backward phases
    Matrix grad_weights_;
    Matrix grad_bias_;
    void initialize_parameters();
};