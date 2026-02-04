#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <span>
#include <cstddef>
#include <algorithm>
#include "types_constants.hpp"

/**
 * @brief Normalizes input vectors by scaling pixel values and applying standardization.
 * @param vectors A matrix of input vectors whose elements (except the first bias term)
 *                will be normalized in-place.
 */
void normalize_inputs(Matrix<double>& vectors) {
    double mean = 0;    
    for (std::size_t i = 0; i < vectors.size(); i++) {
        for (std::size_t j = 1; j < DIM_COUNT; j++) { // skipping the first element (bias)
            vectors[i][j] /= 255; // divide by the max value of a pixel

            mean += vectors[i][j];
        }
    }

    mean  /= (vectors.size() * DIM_COUNT);

    double stdev = 0;
    for (std::size_t i = 0; i < vectors.size(); i++) {
        for (std::size_t j = 1; j < DIM_COUNT; j++) { // skipping the first element (bias)
            double diff = (vectors[i][j] - mean);
            stdev += diff * diff;
        }
    }

    stdev  /= (vectors.size() * DIM_COUNT);
    stdev = sqrt(stdev);

    for (std::size_t i = 0; i < vectors.size(); i++) {
        for (std::size_t j = 1; j < DIM_COUNT; j++) { // skipping the first element (bias)
            vectors[i][j] = (vectors[i][j] - mean) / stdev; 
        }
    }
}

/**
 * @brief Loads input vectors from a CSV file and normalizes them.
 * @param path The path to the file containing comma-separated pixel values.
 * @param num_vectors The expected number of vectors to reserve space for.
 * @return A matrix of normalized input vectors with a leading bias term.
 */
Matrix<double> load_vectors(const std::string& path, std::size_t num_vectors) {
    Matrix<double> vectors;
    vectors.reserve(num_vectors);

    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> vec = {1};
        vec.reserve(DIM_COUNT + 1);

        std::stringstream vector_str(line);
        std::string num_str;

        while (std::getline(vector_str, num_str, ',')) {
            vec.push_back(std::stoi(num_str));
        }

        vectors.push_back(std::move(vec));
    }

    normalize_inputs(vectors);
    return vectors;
}

/**
 * @brief Loads integer labels from a file.
 * @param path The path to the file containing one label per line.
 * @param num_labels The expected number of labels to reserve space for.
 * @return A vector containing all loaded labels.
 */
std::vector<int> load_labels(const std::string& path, std::size_t num_labels) {
    std::vector<int> labels;
    labels.reserve(num_labels);

    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        labels.push_back(std::stoi(line));
    }

    return labels;
}


/**
 * @brief Iterator for traversing a container in fixed-size batches.
 */
template <typename Container>
class BatchIterator {
private:
    const Container* data;
    size_t batch_size;
    size_t index;

public:
    using ValueType = typename Container::value_type;

    /**
     * @brief Constructs a batch iterator.
     * @param data Pointer to the container being iterated.
     * @param batch_size Number of elements per batch.
     * @param index Starting index within the container.
     */
    BatchIterator(const Container* data, size_t batch_size, size_t index)
        : data(data), batch_size(batch_size), index(index) {}

    /**
     * @brief Compares two iterators for inequality based on index.
     * @param other The iterator to compare against.
     * @return True if the iterators point to different positions.
     */
    bool operator!=(const BatchIterator& other) const {
        return index != other.index;
    }

    /**
     * @brief Advances the iterator to the next batch.
     * @return Reference to the updated iterator.
     */
    BatchIterator& operator++() {
        index = std::min(index + batch_size, data->size());
        return *this;
    }

    /**
     * @brief Returns a span representing the current batch.
     * @return A span over the current batch of elements.
     */
    std::span<const ValueType> operator*() const {
        size_t end = std::min(index + batch_size, data->size());
        return std::span<const ValueType>(&(*data)[index], end - index);
    }
};


/**
 * @brief Provides a batch-wise view over a container.
 */
template <typename Container>
class BatchView {
private:
    const Container& data;
    size_t batch_size;

public:
    using Iterator = BatchIterator<Container>;

    /**
     * @brief Constructs a batch view over the given container.
     * @param data The container to iterate over in batches.
     * @param batch_size The number of elements per batch.
     */
    BatchView(const Container& data, size_t batch_size) 
        : data(data), batch_size(batch_size) {}

    /**
     * @brief Returns an iterator to the first batch.
     * @return A batch iterator starting at index 0.
     */
    Iterator begin() const { return Iterator(&data, batch_size, 0); }

    /**
     * @brief Returns an iterator to the end position.
     * @return A batch iterator pointing past the last batch.
     */
    Iterator end() const { return Iterator(&data, batch_size, data.size()); }
};


/**
 * @brief Provides k-fold cross-validation views over a dataset and its labels.
 */
template<typename Container>
class KFoldView {
public:
    const Container& data;
    const std::vector<int>& labels;
    size_t k;
    size_t fold_size;

    /**
     * @brief Constructs a k-fold view over the dataset.
     * @param data The full dataset container.
     * @param labels The corresponding labels for the dataset.
     * @param k The number of folds to split the data into.
     */
    KFoldView(const Container& data,
              const std::vector<int>& labels,
              size_t k)
        : data(data),
          labels(labels),
          k(k),
          fold_size(data.size() / k)
    {}

    /**
     * @brief Retrieves the validation subset for a given fold.
     * @param fold_index The index of the fold to use as validation.
     * @param val_data Output span receiving the validation data slice.
     * @param val_labels Output span receiving the validation label slice.
     */
    void get_validation(size_t fold_index,
                        std::span<const typename Container::value_type>& val_data,
                        std::span<const int>& val_labels) const 
    {
        size_t N = data.size();
        size_t start = fold_index * fold_size;
        size_t end   = (fold_index == k - 1 ? N : start + fold_size);

        val_data   = std::span<const typename Container::value_type>(&data[start], end - start);
        val_labels = std::span<const int>(&labels[start], end - start);
    }

    /**
     * @brief Retrieves the training subsets for a given fold.
     * @param fold_index The index of the fold used as validation (excluded from training).
     * @param top_data Output span receiving the data preceding the validation block.
     * @param top_labels Output span receiving the labels preceding the validation block.
     * @param bottom_data Output span receiving the data following the validation block.
     * @param bottom_labels Output span receiving the labels following the validation block.
     */
    void get_training(size_t fold_index,
                      std::span<const typename Container::value_type>& top_data,
                      std::span<const int>& top_labels,
                      std::span<const typename Container::value_type>& bottom_data,
                      std::span<const int>& bottom_labels) const 
    {
        size_t N = data.size();
        size_t start = fold_index * fold_size;
        size_t end   = (fold_index == k - 1 ? N : start + fold_size);

        // Training = everything except [start, end)
        top_data    = std::span<const typename Container::value_type>(&data[0], start);
        top_labels  = std::span<const int>(&labels[0], start);

        bottom_data   = std::span<const typename Container::value_type>(&data[end], N - end);
        bottom_labels = std::span<const int>(&labels[end], N - end);
    }
};

/**
 * @brief Exports integer results to a file.
 * @param path The output file path.
 * @param results The list of integer results to write.
 */
void export_results(const std::string& path, const std::vector<int>& results) {
    std::ofstream file(path);

    for (const int& label : results) {
        file << label << "\n";
    }
}
