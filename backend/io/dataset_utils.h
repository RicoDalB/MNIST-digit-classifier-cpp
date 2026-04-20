#pragma once

#include <vector>

#include "mnist_reader.h"

struct DatasetSplit {
    std::vector<MNISTSample> train;
    std::vector<MNISTSample> validation;
};

namespace DatasetUtils {
    DatasetSplit train_validation_split(
        const std::vector<MNISTSample>& samples,
        double train_ratio,
        unsigned int seed = 42
    );
}