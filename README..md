# MNIST Digit Classifier in C++

A from-scratch MNIST digit classifier built in C++ with an interactive browser frontend.  
The project combines a neural network backend for training and inference with a simple canvas-based UI that lets users draw digits and receive predictions in real time.

This project is designed to be:
- educational
- demo-friendly
- portfolio-ready
- modular enough to extend later

## Overview

The backend implements a feedforward neural network from scratch in C++ for classifying handwritten digits from the MNIST dataset.  
The frontend provides a drawing canvas where users can sketch a digit, send it to the backend, and view the predicted class along with confidence scores.

The goal is not just to train a model, but to build a complete interactive system:
- train on MNIST
- save and load weights
- expose a prediction API
- visualize predictions in the browser

## Features

- Feedforward neural network implemented from scratch in C++
- MNIST training and evaluation pipeline
- Save and load pretrained weights
- Interactive browser canvas for drawing digits
- Real-time digit prediction
- Confidence scores for all 10 classes
- Modular project structure with separated backend and frontend
- Easy to extend with charts, history, or better architectures

## Project Structure

```text
MNIST/
├── backend/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── api/
│   │   ├── api_server.cpp
│   │   └── api_server.h
│   ├── nn/
│   │   ├── neural_network.cpp
│   │   ├── neural_network.h
│   │   ├── layer.cpp
│   │   ├── layer.h
│   │   ├── activations.cpp
│   │   ├── activations.h
│   │   ├── loss.cpp
│   │   └── loss.h
│   ├── math/
│   │   ├── matrix.cpp
│   │   └── matrix.h
│   ├── io/
│   │   ├── mnist_reader.cpp
│   │   ├── mnist_reader.h
│   │   ├── weights_io.cpp
│   │   └── weights_io.h
│   └── utils.h
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── pretrained/
│   ├── hidden.csv
│   └── output.csv
│
├── outputs/
│   ├── loss.csv
│   ├── accuracy.csv
│   └── confusion_matrix.csv
│
├── README.md
├── LICENSE
├── .gitignore
└── build.sh
```

## Architecture

### Backend
The backend is responsible for:
- loading the MNIST dataset
- training the neural network
- evaluating performance
- saving and loading model weights
- serving predictions to the frontend

### Frontend
The frontend is responsible for:
- collecting drawn digit input from the user
- converting the drawing into a format compatible with MNIST
- sending the processed image to the backend
- displaying the predicted digit and confidence scores

## Model

This project uses a simple multilayer perceptron for digit classification.

Example architecture:

- Input: `784` neurons `(28 × 28 pixels flattened)`
- Hidden layer 1: `128` neurons
- Hidden layer 2: `64` neurons
- Output: `10` neurons `(digits 0–9)`

Recommended choices:
- Hidden activations: ReLU
- Output activation: Softmax
- Loss: Cross-entropy
- Optimizer: SGD or mini-batch gradient descent

## Dataset

The project uses the MNIST handwritten digit dataset.

Each sample consists of:
- a grayscale `28 × 28` image
- a label from `0` to `9`

You can store the dataset as:
- CSV files (`train.csv`, `test.csv`)
- or IDX files if you later want a more standard loader

For a first version, CSV is easier to debug.

## How It Works

1. The backend loads training and test data.
2. The neural network is trained on MNIST.
3. Learned weights are saved into the `pretrained/` directory.
4. When the backend starts in inference mode, it loads the saved weights.
5. The frontend lets the user draw a digit on a canvas.
6. The drawing is resized and normalized to match MNIST format.
7. The processed input is sent to the backend through an API request.
8. The backend runs inference and returns probabilities for digits `0–9`.
9. The frontend displays the final prediction and confidence scores.

## Build and Run

### Prerequisites

- C++17 or newer
- CMake
- A C++ compiler such as `g++`
- A local server for the frontend, or any simple static file server

## Build Backend

```bash
cd backend
mkdir build
cd build
cmake ..
make
```

## Run Backend

Depending on your implementation, `main.cpp` can support:
- training mode
- evaluation mode
- prediction server mode

Example:

```bash
./mnist_backend
```

## Run Frontend

From the `frontend/` folder, serve the files with a simple local server.

Using Python:

```bash
cd frontend
python -m http.server 3000
```

Then open:

```text
http://localhost:3000
```

## Inference Workflow

A typical inference workflow is:

- launch backend server
- open frontend in browser
- draw a digit on the canvas
- click predict
- receive predicted digit and confidence scores

## Outputs

During training, the project can generate:

- `outputs/loss.csv` for training loss
- `outputs/accuracy.csv` for training or validation accuracy
- `outputs/confusion_matrix.csv` for evaluation analysis

These files can be used later for plotting charts.

## Future Improvements

Possible extensions:

- mini-batch training
- dropout
- better preprocessing for canvas input
- confusion matrix visualization in the frontend
- prediction history
- confidence bar charts
- support for convolutional neural networks
- model comparison between MLP and CNN

## Why This Project Is Interesting

This project is useful for learning because it combines:
- low-level neural network implementation
- matrix math and backpropagation
- data preprocessing
- model serialization
- API communication
- frontend visualization

It is a strong project for:
- coursework
- mini-projects
- portfolios
- interview discussion

## License

This project is licensed under the terms of the LICENSE file in this repository.

### Weight Initialization

The network weights are initialized using Xavier uniform initialization to improve training stability across layers. For a layer with `Ni` inputs and `No` outputs, weights are sampled from:

\[
U\left(-\sqrt{\frac{6}{N_i + N_o}}, \sqrt{\frac{6}{N_i + N_o}}\right)
\]

Biases are initialized to zero.

### Output Layer and Classification

For MNIST digit classification, the final layer produces raw class scores (logits). These logits are converted into probabilities using softmax during prediction. For training, the model is intended to use a classification-oriented loss such as cross-entropy with one-hot encoded targets.

## Extension

Use the Best preactive view in cousrse, like parameter initialization and adam ecc..
Allow change activation function via api 


