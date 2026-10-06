#pragma once

#include <vector>
#include <array>

/**
 * @brief Alias for a 2D matrix stored as a vector of vectors.
 * @tparam T Element type.
 */
template<typename T>
using Matrix = std::vector<std::vector<T>>;

/**
 * @brief Concept representing a matrix-like type providing indexed access and size.
 * @tparam T The type to test.
 */
template <typename T>
concept MatrixLike = requires(T m, size_t i, size_t j) {
    { m[i][j] } -> std::convertible_to<double>;
    { m.size() } -> std::convertible_to<size_t>;
};

/**
 * @brief Concept representing a vector-like type providing indexed access and size.
 * @tparam T The type to test.
 */
template <typename T>
concept VectorLike = requires(T v, size_t i) {
    { v[i] } -> std::convertible_to<double>; // element access
    { v.size() } -> std::convertible_to<size_t>; // size
};

/**
 * @brief Activation function types used in neural network layers.
 */
enum struct Activation{
    SOFTMAX,
    RELU
};


constexpr std::size_t BATCH_SIZE = 128;
constexpr std::size_t NUM_TRAIN_VECTORS = 60000;
constexpr std::size_t NUM_TEST_VECTORS = 10000;
constexpr std::size_t DIM_COUNT = 784;
constexpr std::size_t NUM_CLASSES = 10;


constexpr double DROPOUT_PROB = 0.2;
constexpr double LEARNING_RATE_START = 0.00005;
constexpr double LEARNING_RATE_MIN = 1e-6;
constexpr double LEARNING_RATE_MAX = 1e-3;
constexpr double L2_LAMBDA = 0.0005;
constexpr double MOMENTUM_BETA = 0.9;
constexpr double DECAY_BETA = 0.9;
constexpr double EPSILON = 1e-8;
constexpr std::size_t NUM_FOLDS = 8;
constexpr std::size_t EPOCHS = 50;