#include "trainer.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "../io/mnist_reader.h"
#include "../io/weights_io.h"
#include "../nn/activations.h"
#include "../nn/loss.h"
#include "../nn/neural_network.h"

namespace {

struct EvaluationMetrics {
    double average_loss;
    double accuracy;
};

int argmax(const Matrix& row_vector) {
    if (row_vector.rows() != 1) {
        throw std::invalid_argument("argmax expects a row vector with shape 1 x N.");
    }

    int best_index = 0;
    double best_value = row_vector.at(0, 0);

    for (int col = 1; col < row_vector.cols(); ++col) {
        if (row_vector.at(0, col) > best_value) {
            best_value = row_vector.at(0, col);
            best_index = col;
        }
    }

    return best_index;
}

EvaluationMetrics evaluate_dataset(
    NeuralNetwork& network,
    const std::vector<MNISTSample>& samples
) {
    if (samples.empty()) {
        throw std::invalid_argument("Evaluation dataset is empty.");
    }

    double total_loss = 0.0;
    int correct_predictions = 0;

    for (const MNISTSample& sample : samples) {
        Matrix logits = network.forward(sample.input);
        Matrix probabilities = Activations::softmax(logits);

        total_loss += Loss::cross_entropy(probabilities, sample.target);

        const int predicted_label = argmax(probabilities);
        if (predicted_label == sample.label) {
            ++correct_predictions;
        }
    }

    EvaluationMetrics metrics{};
    metrics.average_loss =
        total_loss / static_cast<double>(samples.size());
    metrics.accuracy =
        static_cast<double>(correct_predictions) /
        static_cast<double>(samples.size());

    return metrics;
}

}

void Trainer::run() {
    struct TrainingConfig {
        double learning_rate = 0.02;
        int epochs = 20;
        bool shuffle_each_epoch = true;
        bool evaluate_test_each_epoch = true;
        unsigned int random_seed = 42;
    };

    const TrainingConfig config{};

    const int input_size = 784;
    const std::vector<int> hidden_sizes = {128, 64, 32};
    const int output_size = 10;

    std::cout << "Loading training data...\n";
    std::vector<MNISTSample> train_samples = MNISTReader::load_mnist(
        "../data/train-images.idx3-ubyte",
        "../data/train-labels.idx1-ubyte"
    );

    if (train_samples.empty()) {
        throw std::runtime_error("No training samples loaded.");
    }

    std::cout << "Loading test data...\n";
    std::vector<MNISTSample> test_samples = MNISTReader::load_mnist(
        "../data/t10k-images.idx3-ubyte",
        "../data/t10k-labels.idx1-ubyte"
    );

    if (test_samples.empty()) {
        throw std::runtime_error("No test samples loaded.");
    }

    std::cout << "Training samples: " << train_samples.size() << "\n";
    std::cout << "Test samples: " << test_samples.size() << "\n";

    NeuralNetwork network(input_size, hidden_sizes, output_size);
    std::mt19937 rng(config.random_seed);

    std::cout << "Starting training...\n";

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        if (config.shuffle_each_epoch) {
            std::shuffle(train_samples.begin(), train_samples.end(), rng);
        }

        double total_train_loss = 0.0;
        int correct_train_predictions = 0;

        for (const MNISTSample& sample : train_samples) {
            Matrix logits = network.forward(sample.input);
            Matrix probabilities = Activations::softmax(logits);

            const double loss = Loss::cross_entropy(probabilities, sample.target);
            const Matrix grad_loss =
                Loss::softmax_cross_entropy_gradient(probabilities, sample.target);

            const int predicted_label = argmax(probabilities);
            if (predicted_label == sample.label) {
                ++correct_train_predictions;
            }

            network.backward(grad_loss);
            network.update_parameters(config.learning_rate);

            total_train_loss += loss;
        }

        const double average_train_loss =
            total_train_loss / static_cast<double>(train_samples.size());

        const double train_accuracy =
            static_cast<double>(correct_train_predictions) /
            static_cast<double>(train_samples.size());

        if (config.evaluate_test_each_epoch) {
            const EvaluationMetrics test_metrics =
                evaluate_dataset(network, test_samples);

            std::cout << "Epoch " << (epoch + 1)
                      << "/" << config.epochs
                      << " | Train Loss: " << average_train_loss
                      << " | Train Accuracy: " << train_accuracy
                      << " | Test Loss: " << test_metrics.average_loss
                      << " | Test Accuracy: " << test_metrics.accuracy
                      << "\n";
        } else {
            std::cout << "Epoch " << (epoch + 1)
                      << "/" << config.epochs
                      << " | Train Loss: " << average_train_loss
                      << " | Train Accuracy: " << train_accuracy
                      << "\n";
        }
    }

    std::cout << "Computing final metrics...\n";

    const EvaluationMetrics final_train_metrics =
        evaluate_dataset(network, train_samples);
    const EvaluationMetrics final_test_metrics =
        evaluate_dataset(network, test_samples);

    std::cout << "Final Train Loss: " << final_train_metrics.average_loss << "\n";
    std::cout << "Final Train Accuracy: " << final_train_metrics.accuracy << "\n";
    std::cout << "Final Test Loss: " << final_test_metrics.average_loss << "\n";
    std::cout << "Final Test Accuracy: " << final_test_metrics.accuracy << "\n";

    std::cout << "Saving trained model...\n";
    WeightsIO::save_network(network, "../pretrained/mnist_model.txt");
    std::cout << "Saved to ../pretrained/mnist_model.txt\n";
}