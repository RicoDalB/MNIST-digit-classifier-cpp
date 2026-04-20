#include "api_server.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"
#include "../app/predictor.h"
#include "../math/matrix.h"

namespace {
std::vector<double> parse_pixels_csv(const std::string& body) {
    std::vector<double> pixels;
    std::stringstream ss(body);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            pixels.push_back(std::stod(item));
        }
    }

    return pixels;
}

Matrix pixels_to_matrix(const std::vector<double>& pixels) {
    if (pixels.size() != 784) {
        throw std::invalid_argument("Expected 784 pixel values.");
    }

    Matrix input(1, 784);
    for (int i = 0; i < 784; ++i) {
        input.at(0, i) = pixels[i];
    }
    return input;
}

std::string prediction_to_json(const PredictionResult& result) {
    std::ostringstream out;
    out << "{";
    out << "\"predicted_class\":" << result.predicted_class << ",";
    out << "\"confidence\":" << result.confidence << ",";
    out << "\"probabilities\":[";
    for (std::size_t i = 0; i < result.probabilities.size(); ++i) {
        out << result.probabilities[i];
        if (i + 1 < result.probabilities.size()) {
            out << ",";
        }
    }
    out << "]";
    out << "}";
    return out.str();
}
}

void APIServer::run() {
    Predictor predictor;
    predictor.load_model("../pretrained/mnist_model.txt");

    httplib::Server server;

    server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "*"},
        {"Access-Control-Allow-Methods", "POST, GET, OPTIONS"}
    });

    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    server.Post("/predict", [&predictor](const httplib::Request& req, httplib::Response& res) {
        try {
            std::vector<double> pixels = parse_pixels_csv(req.body);
            Matrix input = pixels_to_matrix(pixels);

            PredictionResult result = predictor.predict(input);
            res.set_content(prediction_to_json(result), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            std::string error_json = std::string("{\"error\":\"") + e.what() + "\"}";
            res.set_content(error_json, "application/json");
        }
    });

    std::cout << "API server listening on http://localhost:8080\n";
    server.listen("0.0.0.0", 8080);
}