#include <iostream>
#include <string>

#include "api/api_server.h"
#include "app/predictor.h"
#include "app/trainer.h"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cout << "Usage:\n";
            std::cout << "  ./backend train\n";
            std::cout << "  ./backend serve\n";
            return 0;
        }

        std::string mode = argv[1];

        if (mode == "train") {
            Trainer::run();
        } else if (mode == "serve") {
            APIServer::run();
        } else {
            std::cout << "Unknown mode: " << mode << "\n";
            std::cout << "Use 'train' or 'serve'.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}