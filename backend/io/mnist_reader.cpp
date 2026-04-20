#include "mnist_reader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    int read_big_endian_int(std::ifstream& file) {
        unsigned char bytes[4];

        file.read(reinterpret_cast<char*>(bytes), 4);
        if (!file) {
            throw std::runtime_error("Failed to read 4-byte integer from MNIST file.");
        }

        return (static_cast<int>(bytes[0]) << 24) |
               (static_cast<int>(bytes[1]) << 16) |
               (static_cast<int>(bytes[2]) << 8)  |
               static_cast<int>(bytes[3]);
    }
}

Matrix MNISTReader::one_hot_encode(int label, int num_classes) {
    if (label < 0 || label >= num_classes) {
        throw std::invalid_argument("Label out of range for one-hot encoding.");
    }

    Matrix target(1, num_classes);
    target.fill(0.0);
    target.at(0, label) = 1.0;

    return target;
}

std::vector<MNISTSample> MNISTReader::load_mnist(
    const std::string& images_path,
    const std::string& labels_path
) {
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file.is_open()) {
        throw std::runtime_error("Failed to open images file: " + images_path);
    }

    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labels_path);
    }

    const int image_magic = read_big_endian_int(images_file);
    const int num_images = read_big_endian_int(images_file);
    const int num_rows = read_big_endian_int(images_file);
    const int num_cols = read_big_endian_int(images_file);

    const int label_magic = read_big_endian_int(labels_file);
    const int num_labels = read_big_endian_int(labels_file);

    if (image_magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number: " + std::to_string(image_magic));
    }

    if (label_magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number: " + std::to_string(label_magic));
    }

    if (num_images != num_labels) {
        throw std::runtime_error(
            "Mismatch between number of images and labels: " +
            std::to_string(num_images) + " vs " + std::to_string(num_labels)
        );
    }

    if (num_rows != 28 || num_cols != 28) {
        throw std::runtime_error(
            "Unexpected MNIST image dimensions: " +
            std::to_string(num_rows) + "x" + std::to_string(num_cols)
        );
    }

    std::vector<MNISTSample> samples;
    samples.reserve(num_images);

    const int pixels_per_image = num_rows * num_cols;

    for (int sample_index = 0; sample_index < num_images; ++sample_index) {
        unsigned char label_byte;
        labels_file.read(reinterpret_cast<char*>(&label_byte), 1);
        if (!labels_file) {
            throw std::runtime_error(
                "Failed to read label for sample " + std::to_string(sample_index)
            );
        }

        int label = static_cast<int>(label_byte);
        Matrix input(1, pixels_per_image);

        for (int pixel_index = 0; pixel_index < pixels_per_image; ++pixel_index) {
            unsigned char pixel_byte;
            images_file.read(reinterpret_cast<char*>(&pixel_byte), 1);
            if (!images_file) {
                throw std::runtime_error(
                    "Failed to read pixel data for sample " + std::to_string(sample_index)
                );
            }

            input.at(0, pixel_index) = static_cast<double>(pixel_byte) / 255.0;
        }

        Matrix target = MNISTReader::one_hot_encode(label);
        samples.push_back({input, target, label});
    }

    return samples;
}