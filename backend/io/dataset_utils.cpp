#include "dataset_utils.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

DatasetSplit DatasetUtils::train_validation_split(
    const std::vector<MNISTSample>& samples,
    double train_ratio,
    unsigned int seed
) {
    if (samples.empty()) {
        throw std::invalid_argument("Cannot split an empty dataset.");
    }

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("train_ratio must be between 0.0 and 1.0.");
    }

    std::vector<MNISTSample> shuffled_samples = samples;

    std::mt19937 rng(seed);
    std::shuffle(shuffled_samples.begin(), shuffled_samples.end(), rng);

    const std::size_t train_count =
        static_cast<std::size_t>(train_ratio * shuffled_samples.size());

    if (train_count == 0 || train_count >= shuffled_samples.size()) {
        throw std::runtime_error("Split produced an invalid train/validation size.");
    }

    DatasetSplit split;

    split.train = std::vector<MNISTSample>(
        shuffled_samples.begin(),
        shuffled_samples.begin() + train_count
    );

    split.validation = std::vector<MNISTSample>(
        shuffled_samples.begin() + train_count,
        shuffled_samples.end()
    );

    return split;
}