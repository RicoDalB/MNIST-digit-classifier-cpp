#pragma once

#include <string>
#include <vector>

#include "../math/matrix.h"

struct MNISTSample {
    Matrix input;   // Shape: 1 x 784
    Matrix target;  // Shape: 1 x 10
    int label;      // Original digit label
};

namespace MNISTReader {
    std::vector<MNISTSample> load_mnist(
        const std::string& images_path,
        const std::string& labels_path
    );

    Matrix one_hot_encode(int label, int num_classes = 10);
}