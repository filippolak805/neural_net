#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <functional>
#include <cmath>
#include <random>
#include <span>
#include "types_constants.hpp"
#include "math_utils.hpp"

/**
 * @brief Applies the ReLU activation function element-wise.
 * @param outputs The output vector to write activated values into.
 * @param potentials The input vector of pre-activation values.
 */
void relu(std::vector<double>& outputs, const std::vector<double>& potentials) {
    for (std::size_t i = 0; i < potentials.size(); i++) {
        outputs[i] = potentials[i] > 0 ? potentials[i] : 0;
    }
}

/**
 * @brief Applies the softmax activation function element-wise.
 * @param outputs The output vector to write normalized probabilities into.
 * @param potentials The input vector of pre-activation values.
 */
void softmax(std::vector<double>& outputs, const std::vector<double>& potentials) {
    double max_val = *std::max_element(potentials.begin(), potentials.end());

    double total = 0.0;
    for (size_t i = 0; i < potentials.size(); i++) {
        outputs[i] = std::exp(potentials[i] - max_val);
        total += outputs[i];
    }

    for (size_t i = 0; i < potentials.size(); i++) {
        outputs[i] /= total;
    }
}

/**
 * @brief Represents a neural network layer with activations, dropout, and gradients.
 */
struct Layer {
    std::size_t params_in; // number of weights inside a neuron
    std::size_t params_out; // number of neurons in this layer (dimension of the input to the next)
    std::size_t max_batch_size;
    std::size_t batch_size;
    std::size_t dropout_prob = DROPOUT_PROB;

    Matrix<double> weights;             // [params_out][params_in]
    Matrix<double> potentials;          // [batch_size][params_out]
    Matrix<double> outputs;             // [batch_size][params_out]
    Matrix<double> weight_gradients;    // [params_out][params_in]
    Matrix<double> output_gradients;    // [batch_size][params_out]
    Matrix<double> rms_cache;           // [params_out][params_in]
    Matrix<double> momentums;           // [params_out][params_in]
    Matrix<double> dropout_mask;        // [batch_size][params_out]

    Activation activation_type;
    std::function<void(std::vector<double>&, std::vector<double>&)> activation;

    bool training = true;

    /**
     * @brief Constructs a neural network layer with initialized weights.
     * @param in Number of input parameters per neuron.
     * @param out Number of output neurons.
     * @param act Activation function type.
     * @param max_batch_size Maximum batch size supported.
     */
    Layer(uint32_t in, uint32_t out, Activation act, std::size_t max_batch_size) 
            : params_in(in), params_out(out), max_batch_size(max_batch_size), activation_type(act)
    {
        initialize();
        batch_size = max_batch_size;

        switch (activation_type)
        {
        case Activation::SOFTMAX:
            activation = softmax;
            break;
        case Activation::RELU:
            activation = relu;
            break;
        default:
            break;
        }
    }

    /**
     * @brief Initializes weights, gradients, caches, and dropout masks.
     */
    void initialize() {
        weights.assign(params_out, std::vector<double>(params_in, 0.0));
        weight_gradients.assign(params_out, std::vector<double>(params_in, 0.0));
        rms_cache.assign(params_out, std::vector<double>(params_in, 0.0));
        momentums.assign(params_out, std::vector<double>(params_in, 0.0));
        output_gradients.assign(max_batch_size, std::vector<double>(params_out, 0.0));
        outputs.assign(max_batch_size, std::vector<double>(params_out, 0.0));
        potentials.assign(max_batch_size, std::vector<double>(params_out, 0.0));
        dropout_mask.assign(max_batch_size, std::vector<double>(params_out, 0.0));

        double stddev = std::sqrt(2.0 / params_in);
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, stddev);

        for (auto& neuron : weights)
            for (double& w : neuron)
                w = dist(gen);
    };

    /**
     * @brief Performs a forward pass through the layer.
     * @param input The input batch matrix.
     */
    template <MatrixLike I>
    void forward(const I& input) {
        potentials.resize(batch_size, std::vector<double>(params_out));
        outputs.resize(batch_size, std::vector<double>(params_out));
        output_gradients.resize(batch_size, std::vector<double>(params_out));

        mat_mul(weights, input, potentials);

        for (std::size_t i = 0; i < potentials.size(); i++)
            activation(outputs[i], potentials[i]);


        if (training && activation_type != Activation::SOFTMAX) {
            std::mt19937 gen(42);
            std::bernoulli_distribution dist(1.0 - dropout_prob);

            for (std::size_t b = 0; b < batch_size; b++) {
                for (std::size_t j = 0; j < params_out; j++) {
                    dropout_mask[b][j] = dist(gen) ? 1.0 : 0.0;
                    outputs[b][j] *= dropout_mask[b][j];
                    outputs[b][j] /= (1.0 - dropout_prob);  // inverted dropout scaling
                }
            }
        } else {
            // No dropout at inference
            for (std::size_t b = 0; b < batch_size; b++)
                for (std::size_t j = 0; j < params_out; j++)
                    dropout_mask[b][j] = 1.0;
        }
    }
};

/**
 * @brief Represents a neural network composed of multiple layers.
 */
struct Net {
    std::vector<Layer> layers;
    std::size_t max_batch_size = BATCH_SIZE;
    std::size_t batch_size = max_batch_size;
    bool training = true;

    /**
     * @brief Constructs an empty network with reserved layer capacity.
     */
    Net() {
        layers.reserve(10);
    }

    /**
     * @brief Copy-constructs a network from another instance.
     * @param other The network to copy.
     */
    Net(const Net& other)
        : layers(other.layers),                // deep copy via Layer's copy ctor
        max_batch_size(other.max_batch_size),
        batch_size(other.batch_size),
        training(other.training)
    {}

    /**
     * @brief Assigns another network to this one.
     * @param other The network to copy.
     * @return Reference to this network.
     */
    Net& operator=(const Net& other) {
        if (this != &other) {
            layers = other.layers;
            max_batch_size = other.max_batch_size;
            batch_size = other.batch_size;
            training = other.training;
        }
    
        return *this;
    }

    /**
     * @brief Adds a new layer to the network.
     * @param params_in Number of input parameters per neuron.
     * @param params_out Number of neurons in the new layer.
     * @param act Activation function type.
     */
    void add_layer(uint32_t params_in, uint32_t params_out, Activation act) {
        layers.emplace_back(params_in, params_out, act, max_batch_size);
    }

    /**
     * @brief Resets all network weights to zero.
     */
    void reset_weights() {
        for (auto& layer : layers)
            layer.weights.assign(layer.params_out, std::vector<double>(layer.params_in, 0.0));
    }

    /**
     * @brief Adds weights from another network into this one.
     * @param other The network providing weights to add.
     */
    void add_weights_from(const Net& other) {
        for (size_t i = 0; i < layers.size(); i++)
            mat_add(layers[i].weights, other.layers[i].weights);
    }

    /**
     * @brief Scales all weights in the network by a constant factor.
     * @param k The scaling factor.
     */
    void scale_weights(double k) {
        for (auto& layer : layers)
            mat_mul_scalar(layer.weights, k);
    }

    /**
     * @brief Performs a forward pass through all layers of the network.
     * @param input The input batch matrix.
     */
    template<MatrixLike I>
    void forward(const I& input) {
        batch_size = input.size();
        layers[0].batch_size = batch_size;
        layers[0].training = training;
        layers[0].forward(input);

        for (uint32_t i = 1; i < layers.size(); i++) {
            layers[i].batch_size = layers[i - 1].batch_size;
            layers[i].training = training;
            layers[i].forward(layers[i - 1].outputs);
        }
    }

    /**
     * @brief Predicts class labels for an input dataset.
     * @param input The full dataset to run inference on.
     * @return A vector of predicted integer labels.
     */
    template <MatrixLike I>
    std::vector<int> predict(const I& input) {
        std::vector<int> predictions;

        training = false;

        BatchView batches(input, batch_size);
        for (auto batch : batches) {
            forward(batch); // Forward pass

            for (const auto& output_vector : layers.back().outputs) {
                // Find index of maximum probability
                auto max_it = std::max_element(output_vector.begin(), output_vector.end());
                int predicted_label = static_cast<int>(std::distance(output_vector.begin(), max_it));
                predictions.push_back(predicted_label);
            }
        }

        training = true;
        return predictions;
    }
};
