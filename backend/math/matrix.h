#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

// Matrix class -> stores a 2D matrix using a flat 1D vector internally
class Matrix{
public:
    Matrix(int rows, int cols);
    int rows() const;
    int cols() const;
    double& at(int row, int col); // read and modify an element
    double at(int row, int col) const; // read element
    void fill(double value);
    Matrix transponse() const; //this return a matrix that is the transponse of this one so dont modify directly this
    void print() const; 
    static Matrix add(const Matrix& a, const Matrix& b);
    static Matrix multiply(const Matrix& a, const Matrix& b);

private:
    int rows_;
    int cols_;
    std::vector<double> data_;
    int index(int rows, int cols) const; // convert 2D position in 1D index

};