#pragma once

#include <omp.h>
#include <vector>
#include <span>
#include <cmath>
#include <chrono>
#include "net.hpp"
#include "types_constants.hpp"

/**
 * @brief Optimizer implementing RMSProp with momentum, k-fold training, and adaptive learning rate.
 */
template<MatrixLike ID, VectorLike LD>
struct Optimizer {
    Net& net;
    ID& input_data;
    LD& label_data;

    double learning_rate = LEARNING_RATE_START;
    double l2_lambda = L2_LAMBDA;
    double momentum_beta = MOMENTUM_BETA;
    double decay_beta = DECAY_BETA;
    double epsilon = EPSILON;
    
    std::size_t epochs = EPOCHS;
    std::vector<double> loss_history;

    /**
     * @brief Constructs the optimizer with network, input data, and label data.
     * @param n Reference to the neural network.
     * @param i Reference to the input dataset.
     * @param l Reference to the label dataset.
     */
    Optimizer(Net& n, ID& i, LD& l) : net(n), input_data(i), label_data(l) {}

    /**
     * @brief Computes cross-entropy loss for a batch of outputs and labels.
     * @param outputs The model output probabilities.
     * @param labels The integer class labels.
     * @return The average cross-entropy loss for the batch.
     */
    template<MatrixLike O, VectorLike L>
    double cross_entropy(const O& outputs, const L& labels) {
        const int batch_size = outputs.size();
        double loss = 0.0;

        for (int b = 0; b < batch_size; ++b) {
            int y = labels[b];
            double p = outputs[b][y];

            // prevent log(0)
            const double eps = 1e-15;
            loss += -std::log(std::max(p, eps));
        }

        return loss / batch_size;
    }

    /**
     * @brief Performs backpropagation and computes gradients for all layers.
     * @param inputs The input batch.
     * @param labels The corresponding labels for the batch.
     * @param local_net The network on which to compute gradients.
     */
    template<MatrixLike I, VectorLike L>
    void backpropagation(const I& inputs, const L& labels, Net& local_net) {
        int num_layers = local_net.layers.size();
        int batch_size = labels.size();
        int num_classes = local_net.layers.back().params_out;

        for (auto& layer : local_net.layers) {
            layer.weight_gradients.assign(layer.params_out, std::vector<double>(layer.params_in, 0.0));
            layer.output_gradients.assign(layer.batch_size, std::vector<double>(layer.params_out, 0.0));
        }

        Layer& last = local_net.layers.back();
        Layer& last_hidden = local_net.layers[num_layers - 2];
        for (int k = 0; k < batch_size; k++)
        {
            // ---------- STEP 1: softmax output gradient ----------
            for (int c = 0; c < num_classes; c++)
                last.output_gradients[k][c] = last.outputs[k][c] - (labels[k] == c ? 1.0 : 0.0);

            for (std::size_t j = 0; j < last.params_out; j++)       // output neurons
                for (std::size_t i = 0; i < last.params_in; i++)   // inputs to output layer (last hidden layer)
                    last.weight_gradients[j][i] += last.output_gradients[k][j] * last_hidden.outputs[k][i];

            for (std::size_t j = 0; j < last_hidden.params_out; j++) {
                for (std::size_t r = 0; r < last.params_out; r++) {
                    last_hidden.output_gradients[k][j] += last.dropout_mask[k][r] * last.weights[r][j] *
                                                        (last.potentials[k][j] > 0 ? 1 : 0) * // derivative of relu 
                                                        last.output_gradients[k][r];
                }
            }

            // ---------- STEP 2: backprop through layers (just relu) ----------
            for (int l = num_layers - 2; l > 0; l--)
            {
                Layer& layer = local_net.layers[l];
                Layer& prev  = local_net.layers[l - 1];

                // (A) accumulate weight gradients
                for (std::size_t j = 0; j < layer.params_out; j++) {
                    for (std::size_t i = 0; i < layer.params_in; i++) {
                        layer.weight_gradients[j][i] += layer.dropout_mask[k][j] * layer.output_gradients[k][j] * 
                                                        (layer.potentials[k][j] > 0 ? 1 : 0) * // derivative of relu
                                                        prev.outputs[k][i];
                    }
                }

                // (B) compute output gradients for previous layer
                for (std::size_t j = 0; j < prev.params_out; j++) {
                    for (std::size_t r = 0; r < layer.params_out; r++) {
                        prev.output_gradients[k][j] += layer.dropout_mask[k][r] * layer.weights[r][j] * 
                                                    (layer.potentials[k][r] > 0 ? 1 : 0) * // derivative of relu
                                                    layer.output_gradients[k][r];
                    }
                }
            }

            // (A) accumulate weight gradients for the first layer
            Layer& first = local_net.layers.front();
            for (std::size_t j = 0; j < first.params_out; j++) {
                for (std::size_t i = 0; i < first.params_in; i++) {
                    first.weight_gradients[j][i] += first.dropout_mask[k][j] * first.output_gradients[k][j] * 
                                                    (first.potentials[k][j] > 0 ? 1 : 0) * // derivative of relu
                                                    inputs[k][i];
                }
            }
        }

        // normalizing by batch size
        for (auto& layer : local_net.layers)
            mat_mul_scalar(layer.weight_gradients, 1.0 / local_net.batch_size);
    }

    /**
     * @brief Applies RMSProp with momentum to update network weights.
     * @param inputs The input batch.
     * @param labels The corresponding labels for the batch.
     * @param local_net The network whose weights are being updated.
     */
    template<MatrixLike I, VectorLike L>
    void RMSProp_momentum(const I& inputs, const L& labels, Net& local_net) {
        // compute the gradients with backpropagation
        backpropagation(inputs, labels, local_net);

        for (auto& layer : local_net.layers) {
            // RMSProp + momentum
            for (size_t j = 0; j < layer.params_out; j++) {
                for (size_t i = 0; i < layer.params_in; i++) {
                    double g = layer.weight_gradients[j][i];

                    // l2 regularization
                    g += 2.0 * l2_lambda * layer.weights[j][i];

                    // update rms cache
                    layer.rms_cache[j][i] = decay_beta * layer.rms_cache[j][i] +
                                            (1.0 - decay_beta) * (g * g);

                    // update momentums
                    layer.momentums[j][i] = momentum_beta * layer.momentums[j][i] +
                                            g / (std::sqrt(layer.rms_cache[j][i]) + epsilon);
                }
            }

            // rms_prop step
            mat_add(layer.weights, layer.momentums, -learning_rate);
        }
    }

    /**
     * @brief Computes a scheduled learning rate based on loss history.
     * @param lr_max The maximum allowed learning rate.
     * @param lr_min The minimum allowed learning rate.
     * @return The adjusted learning rate.
     */
    double learning_rate_schedule(double lr_max=LEARNING_RATE_START, double lr_min=LEARNING_RATE_MIN) {
        const auto& h = loss_history;

        if (h.size() < 2)
            return lr_max;  // first epoch, use max LR

        double first = h.front();
        double prev = h[h.size() - 2];
        double curr  = h.back();

        // relative improvement since the beginning
        double change = (prev - curr) / (first - curr);  // 0 if no improvement, 1 if perfect

        // gentle scaling of LR
        double alpha = 0.1; // how strongly LR decays over the trend
        double lr = learning_rate - alpha * (1.0 - change) * (learning_rate - lr_min);

        // clamp
        if (lr > lr_max) lr = lr_max;
        if (lr < lr_min) lr = lr_min;

        return lr;
    }

    /**
     * @brief Trains the network on the full dataset.
     * @param max_minutes Maximum training duration.
     */
    void train(double max_minutes = 9.0) {
        auto start_time = std::chrono::steady_clock::now();
        auto max_duration = std::chrono::minutes(static_cast<int>(max_minutes));

        for (std::size_t epoch = 0; epoch < epochs; epoch++) {

            learning_rate = learning_rate_schedule();
            std::cout << "learning rate: " << learning_rate << "\n";

            BatchView batches_vectors(input_data, net.batch_size);
            BatchView batches_labels(label_data, net.batch_size);

            auto vec_it  = batches_vectors.begin();
            auto lab_it  = batches_labels.begin();

            for (; vec_it != batches_vectors.end(); ++vec_it, ++lab_it) {
                auto vec = *vec_it;
                auto lab = *lab_it;

                net.forward(vec);                 
                RMSProp_momentum(vec, lab, net);
            }
            
            BatchView val_vecs(input_data, net.batch_size);
            BatchView val_labs(label_data, net.batch_size);

            double loss = 0.0;
            std::size_t batches = 0;

            auto vv = val_vecs.begin();
            auto ll = val_labs.begin();
            for (; vv != val_vecs.end(); ++vv, ++ll) {
                auto vec = *vv;
                auto lab = *ll;

                net.training = false;
                net.forward(vec);
                net.training = true;

                loss += cross_entropy(net.layers.back().outputs, lab);
                batches++;
            }

            loss /= batches;
            std::cout << "epoch " << epoch << " loss = " << loss << "\n";

            loss_history.push_back(loss);

            if (std::chrono::steady_clock::now() - start_time > max_duration) {
                std::cout << "Stopped after " << max_minutes << " minutes.\n";
                break;
            }
        }
    }

};