#include "matrix.h"

//Constructor
Matrix::Matrix(int rows, int cols):rows_(rows), cols_(cols), data_(rows * cols, 0.0){
    if(rows <= 0 || cols <= 0){
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
}

int Matrix::rows() const{
    return rows_;
}

int Matrix::cols() const{
    return cols_;
}

// Func that maps (row, col) to vector index
int Matrix::index(int row, int col) const{
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of range.");
    }
    return row * cols_ + col;
}

double& Matrix::at(int row, int col){
    return data_[index(row, col)];
}

double Matrix::at(int row, int col) const{
    return data_[index(row, col)]; 
}

void Matrix::fill(double value){
    for(double& x: data_){
        x = value;
    }
}

Matrix Matrix::transponse() const{
    Matrix result(cols_, rows_); // Create a matrix result with cols_ rows_ (inverted becouse transpose)

    for(int r = 0; r < rows_; r++){
        for(int c = 0; c < cols_; c++){
            result.at(c, r) = at(r, c);
        }
    }
    return result;
}

void Matrix::print() const{
    for(int r = 0; r < rows_; r++){
        for(int c = 0; c < cols_; c++){
            std::cout << at(r, c) << " ";
        }
        std::cout << "\n";
    }
}

// function for add two matrix
Matrix Matrix::add(const Matrix& a, const Matrix& b){
    if (a.rows() != b.rows() || a.cols() != b.cols()){
        throw std::invalid_argument("Matrix addition requires same dimensions.");
    }
    Matrix result(a.rows(), a.cols()); // Create result matrix
    for(int r = 0; r < a.rows(); r++){
        for(int c = 0; c < a.cols(); c++){
            result.at(r, c) = a.at(r, c) + b.at(r, c);
        }
    }
    return result;
}

// Function for dot product
Matrix Matrix::multiply(const Matrix& a, const Matrix& b){
    if (a.cols() != b.rows()){
        throw std::invalid_argument("Matrix dimension invalid_.");
    }
    Matrix result(a.rows(), b.cols()); // Create result matrix
    for(int r = 0; r < a.rows(); r++){
        for(int c = 0; c < b.cols(); c++){
            double sum = 0.0;
            for(int k = 0; k < a.cols(); k++){
                sum += a.at(r, k) * b.at(k, c);
            }
            result.at(r, c) = sum;
        }
    }
    return result;
}