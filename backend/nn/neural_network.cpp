#include "neural_network.h"
#include <iostream>
#include <stdexcept>
#include "activations.h"
#include "loss.h"

NeuralNetwork::NeuralNetwork(int input_size, const std::vector<int>& hidden_sizes, int output_size):
    input_size_(input_size), output_size_(output_size), hidden_sizes_(hidden_sizes){
        if (input_size <= 0){
            throw std::invalid_argument("Input size must be positive");
        }
        if(output_size <= 0){
            throw std::invalid_argument("Output size must be positive");
        }
        for(int hidden_size: hidden_sizes_){
            if(hidden_size <= 0){
                throw std::invalid_argument("Hidden layer must be positive");
            }
        }
        int previus_size = input_size_;
        // build hidden layers in sequence
        for(int hidden_size : hidden_sizes_){
            layers_.emplace_back(previus_size, hidden_size);
            previus_size = hidden_size;
        }
        // final layer
        layers_.emplace_back(previus_size, output_size_);
    }

Matrix NeuralNetwork::forward(const Matrix& input){
    if(input.rows() != 1 || input.cols() != input_size_){
        throw std::invalid_argument("Input shape not match network input size");
    }
    Matrix current = input;
    for(std::size_t i = 0; i < layers_.size(); i++){
        current = layers_[i].forward(current);
        
        if(i < layers_.size() - 1){
            current = Activations::ReLu(current);
        }
    }
    return current;
}

void NeuralNetwork::backward(const Matrix& grad_loss) {
    if (layers_.empty()) {
        throw std::runtime_error("Network has no layers.");
    }

    Matrix current_grad = grad_loss;

    // Start from the output layer and move backward
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        current_grad = layers_[i].backward(current_grad);

        // Apply ReLU derivative for hidden layers only
        // There is no activation after the final output layer
        if (i > 0) {
            current_grad = Activations::relu_derivative(
                current_grad,
                layers_[i - 1].last_output()
            );
        }
    }
}

void NeuralNetwork::update_parameters(double learning_rate) {
    for (DenseLayer& layer : layers_) {
        layer.upgrade_parameters(learning_rate);
    }
}

double NeuralNetwork::train_sample(const Matrix& input, const Matrix& target, double learning_rate) {
    Matrix logits = forward(input);
    Matrix probabilities = Activations::softmax(logits);

    double loss = Loss::cross_entropy(probabilities, target);
    Matrix grad_loss = Loss::softmax_cross_entropy_gradient(probabilities, target);

    backward(grad_loss);
    update_parameters(learning_rate);

    return loss;
}

void NeuralNetwork::print_structure() const {
    std::cout << "Neural Network Structure:\n";
    std::cout << "Input size: " << input_size_ << '\n';

    for (std::size_t i = 0; i < hidden_sizes_.size(); ++i) {
        std::cout << "Hidden layer " << i + 1
                  << " size: " << hidden_sizes_[i] << '\n';
    }

    std::cout << "Output size: " << output_size_ << '\n';
    std::cout << "Total dense layers: " << layers_.size() << '\n';
}

std::vector<DenseLayer>& NeuralNetwork::layers() {
    return layers_;
}

const std::vector<DenseLayer>& NeuralNetwork::layers() const {
    return layers_;
}