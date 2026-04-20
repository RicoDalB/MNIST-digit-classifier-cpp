cmake_minimum_required(VERSION 3.10)
project(mnist_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(backend
    main.cpp
    math/matrix.cpp
    nn/layer.cpp
    nn/activations.cpp
    nn/neural_network.cpp
    nn/loss.cpp
    io/mnist_reader.cpp
    io/dataset_utils.cpp
    io/weights_io.cpp
    app/trainer.cpp
    app/predictor.cpp
    api/api_server.cpp
)

target_include_directories(backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)