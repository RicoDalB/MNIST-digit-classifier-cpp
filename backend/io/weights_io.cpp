#include "weights_io.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace WeightsIO {

void save_network(const NeuralNetwork& network, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + file_path);
    }

    const std::vector<DenseLayer>& layers = network.layers();

    file << layers.size() << "\n";

    for (const DenseLayer& layer : layers) {
        file << layer.input_size() << " " << layer.output_size() << "\n";

        const Matrix& weights = layer.weights();
        for (int row = 0; row < weights.rows(); ++row) {
            for (int col = 0; col < weights.cols(); ++col) {
                file << weights.at(row, col);
                if (col + 1 < weights.cols()) {
                    file << " ";
                }
            }
            file << "\n";
        }

        const Matrix& bias = layer.bias();
        for (int col = 0; col < bias.cols(); ++col) {
            file << bias.at(0, col);
            if (col + 1 < bias.cols()) {
                file << " ";
            }
        }
        file << "\n";
    }
}

void load_network(NeuralNetwork& network, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + file_path);
    }

    std::vector<DenseLayer>& layers = network.layers();

    int saved_layer_count = 0;
    file >> saved_layer_count;

    if (!file) {
        throw std::runtime_error("Failed to read saved layer count from file.");
    }

    if (saved_layer_count != static_cast<int>(layers.size())) {
        throw std::runtime_error("Saved model layer count does not match current network.");
    }

    for (int layer_index = 0; layer_index < saved_layer_count; ++layer_index) {
        int saved_input_size = 0;
        int saved_output_size = 0;

        file >> saved_input_size >> saved_output_size;

        if (!file) {
            throw std::runtime_error("Failed to read layer dimensions from file.");
        }

        DenseLayer& layer = layers[layer_index];

        if (saved_input_size != layer.input_size() ||
            saved_output_size != layer.output_size()) {
            throw std::runtime_error("Saved layer dimensions do not match current network structure.");
        }

        Matrix weights(saved_input_size, saved_output_size);
        for (int row = 0; row < saved_input_size; ++row) {
            for (int col = 0; col < saved_output_size; ++col) {
                file >> weights.at(row, col);
                if (!file) {
                    throw std::runtime_error("Failed to read weight value from file.");
                }
            }
        }

        Matrix bias(1, saved_output_size);
        for (int col = 0; col < saved_output_size; ++col) {
            file >> bias.at(0, col);
            if (!file) {
                throw std::runtime_error("Failed to read bias value from file.");
            }
        }

        layer.set_weights(weights);
        layer.set_bias(bias);
    }
}

}