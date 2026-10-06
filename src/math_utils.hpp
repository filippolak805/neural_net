#pragma once

#include <cmath>
#include <iostream>
#include "types_constants.hpp"


/**
 * @brief Provides a transposed view of a matrix-like container.
 * @tparam MatrixLike A type modeling a matrix interface.
 */
template <typename MatrixLike>
struct TransposedView {
    MatrixLike& mat;

    /**
     * @brief Constructs a transposed view of a matrix.
     * @param m The matrix to wrap.
     */
    explicit TransposedView(MatrixLike& m) : mat(m) {}

    /**
     * @brief Accesses an element in the transposed view.
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the transposed element.
     */
    auto& operator()(size_t i, size_t j) {
        return mat[j][i];
    }

    /**
     * @brief Accesses a const element in the transposed view.
     * @param i Row index.
     * @param j Column index.
     * @return Const reference to the transposed element.
     */
    const auto& operator()(size_t i, size_t j) const {
        return mat[j][i];
    }

    /**
     * @brief Returns the number of rows in the transposed matrix.
     * @return Number of rows.
     */
    size_t rows() const { return mat[0].size(); }

    /**
     * @brief Returns the number of columns in the transposed matrix.
     * @return Number of columns.
     */
    size_t cols() const { return mat.size(); }
};

/**
 * @brief Computes the dot product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @return The dot product result.
 */
template<typename T>
T operator*(const std::vector<T>& a, const std::vector<T>& b) {
    T output = 0;
    for (size_t i = 0; i < a.size(); i++) {
        output += a[i] * b[i];
    }
    return output;
}


/**
 * @brief Multiplies a weight matrix by an input matrix.
 * 
 * We assume every inner vector in weights is a row of the matrix,
 * while every inner vector in inputs is a column
 * 
 * @param weights The weight matrix.
 * @param inputs The input matrix.
 * @param result The output matrix to write into.
 */
template<MatrixLike W, MatrixLike I, MatrixLike R>
void mat_mul(const W& weights, const I& inputs, R& result) {
    TransposedView inputs_t(inputs); // We want the inputs as columns

    size_t M = weights.size();
    size_t O = (M > 0) ? weights[0].size() : 0;
    size_t N = (inputs_t.rows() > 0) ? inputs_t.cols() : 0;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            result[j][i] = 0;
            for (size_t k = 0; k < O; k++) {
                result[j][i] += weights[i][k] *  inputs_t(k, j);
            }
        }
    }
}



/**
 * @brief Adds one matrix into another with optional scaling.
 * 
 * We assume every inner vector in weights is a row of the matrix,
 * while every inner vector in inputs is a column
 * 
 * @param result The matrix to accumulate into.
 * @param to_add The matrix being added.
 * @param coef Scaling factor applied to the added matrix.
 */
template<MatrixLike R, MatrixLike A>
void mat_add(R& result, const A& to_add, const double coef=1) {
    size_t M = to_add.size();
    size_t N = (M > 0) ? to_add[0].size() : 0;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            result[i][j] += coef * to_add[i][j];
        }
    }
}


/**
 * @brief Prints a vector to standard output.
 * @param v The vector to print.
 */
template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "( ";
    for (const auto& num : v) {
        std::cout << num << ' ';
    }
    std::cout << ")\n";
}

/**
 * @brief Prints a matrix to standard output.
 * @param a The matrix to print.
 */
template<typename MatrixLike>
void print_matrix(const MatrixLike& a) {
    for (const auto& row : a) {
        print_vector(row);
    }
}

/**
 * @brief Multiplies every element in a matrix by a scalar.
 * @param result The matrix to scale.
 * @param num The scalar multiplier.
 */
template<MatrixLike R>
void mat_mul_scalar(R& result, const double num){
    size_t M = result.size();
    size_t N = (M > 0) ? result[0].size() : 0;

    for (size_t i = 0; i < M; i++){
        for (size_t j = 0; j < N; j++){
            result[i][j] *= num;
        }
    }
}

/**
 * @brief Adds a scalar to every element in a matrix.
 * @param result The matrix to update.
 * @param num The scalar to add.
 */
template<MatrixLike R>
void mat_add_scalar(R& result, const double num){
    size_t M = result.size();
    size_t N = (M > 0) ? result[0].size() : 0;

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            result[i][j] += num;
}
