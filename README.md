# MNIST Digit Classifier in C++

A handwritten digit classifier built **from scratch in C++**, featuring a lightweight browser interface for **interactive, real-time inference**.

This project was designed as a **portfolio piece** to showcase low-level machine learning implementation, model serving, and full-stack integration. Instead of relying on high-level ML frameworks, the neural network logic is implemented manually in C++, then exposed through a local backend that communicates with a canvas-based frontend in the browser.

---

## Features

- Neural network written fully in **C++**
- Trained and evaluated on **MNIST**
- **Pretrained model included** for quick demo use
- Backend server for handling predictions
- Frontend with an interactive drawing canvas
- Real-time predicted class and confidence scores
- Clear separation between training, serving, and user interaction

---

## Results

The current trained model achieves:

- **Final Train Accuracy:** `98.30%`
- **Final Test Accuracy:** `96.93%`

These results show strong performance for a handwritten digit recognizer implemented without external deep learning frameworks.

---

## Tech Stack

- **C++**
- **CMake**
- **HTML / CSS / JavaScript**
- **MNIST**

---

## Project Structure

```text
.
├── backend/
│   ├── api/
│   │   ├── api_server.cpp
│   │   ├── api_server.h
│   │   └── httplib.h
│   ├── app/
│   │   ├── predictor.cpp
│   │   ├── predictor.h
│   │   ├── trainer.cpp
│   │   └── trainer.h
│   ├── io/
│   │   ├── dataset_utils.cpp
│   │   ├── dataset_utils.h
│   │   ├── mnist_reader.cpp
│   │   ├── mnist_reader.h
│   │   ├── weights_io.cpp
│   │   └── weights_io.h
│   ├── math/
│   │   ├── matrix.cpp
│   │   └── matrix.h
│   ├── nn/
│   │   ├── activations.cpp
│   │   ├── layer.cpp
│   │   ├── layer.h
│   │   ├── loss.cpp
│   │   ├── loss.h
│   │   ├── neural_network.cpp
│   │   └── neural_network.h
│   ├── CMakeLists.txt
│   └── main.cpp
├── data/
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── README.md
```
---

## How It Works

The project follows a simple end-to-end workflow:

1. Train a neural network on the MNIST dataset
2. Save the learned weights to disk
3. Start the backend in server mode
4. Launch the frontend in the browser
5. Draw a digit and receive a live prediction

This structure makes it easy to understand the complete pipeline from model creation to deployment-like usage.

---

## Getting Started

### 1. Build the backend

```bash
cd build
cmake ..
make
```

---

## Option A — Try the Pretrained Model

If you want to test the project immediately, you can use the pretrained model already included in the repository.

### Start the backend server

```bash
cd build
./backend serve
```

### Start the frontend

From the project root:

```bash
python -m http.server 5500 -d frontend
```

Then open:

```text
http://localhost:5500
```

You can now draw a digit directly in the browser and see the model’s prediction in real time.

This is the fastest way to explore the project without waiting for training.

---

## Option B — Train the Model Yourself

If you want to reproduce the model from scratch, train it manually and then serve your own trained weights.

### Train the network

```bash
cd build
./backend train
```

After training completes, start the backend server:

```bash
./backend serve
```

In a second terminal, launch the frontend:

```bash
python -m http.server 5500 -d frontend
```

Then open:

```text
http://localhost:5500
```

This path is useful if you want to inspect the full workflow, validate the training process, or experiment with changes to the architecture and parameters.

---

## Quick Demo

### Terminal 1

```bash
cd build
./backend serve
```

### Terminal 2

```bash
python -m http.server 5500 -d frontend
```

Then open the frontend in your browser and draw a digit to test the classifier.

---

## Future Improvements

Some possible next steps for expanding the project:

- support for configurable architectures and hyperparameters
- improved preprocessing for canvas input
- model visualization and training metrics plots
- REST API documentation
- Dockerized setup for easier portability
- comparison with deeper or alternative architectures

