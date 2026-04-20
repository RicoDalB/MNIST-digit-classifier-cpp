#pragma once

#include <string>

#include "../nn/neural_network.h"

namespace WeightsIO {
    void save_network(const NeuralNetwork& network, const std::string& file_path);
    void load_network(NeuralNetwork& network, const std::string& file_path);
}