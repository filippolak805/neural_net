#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "types_constants.hpp"
#include "data.hpp"
#include "math_utils.hpp"
#include "net.hpp"
#include "train.hpp"


int main() {
    auto inputs_train = load_vectors("./data/fashion_mnist_train_vectors.csv", NUM_TRAIN_VECTORS);
    auto labels_train = load_labels("./data/fashion_mnist_train_labels.csv", NUM_TRAIN_VECTORS);

    auto inputs_test = load_vectors("./data/fashion_mnist_test_vectors.csv", NUM_TEST_VECTORS);
    auto labels_test = load_labels("./data/fashion_mnist_test_labels.csv", NUM_TEST_VECTORS);

    Net net;
    net.max_batch_size = BATCH_SIZE;

    net.add_layer(DIM_COUNT + 1, 128, Activation::RELU);
    net.add_layer(128, 32, Activation::RELU);
    net.add_layer(32, 10, Activation::SOFTMAX);

    Optimizer optimizer(net, inputs_train, labels_train);
    optimizer.train();


    auto test_predictions = net.predict(inputs_test);

    double accuracy = 0.0;
    for (std::size_t i = 0; i < test_predictions.size(); i++) {
        if (test_predictions[i] == labels_test[i])
            accuracy++;
    }
    accuracy /= NUM_TEST_VECTORS;
    std::cout << "Accuracy: " << accuracy << "\n";

    export_results("./test_predictions.csv", test_predictions);

    auto train_predictions = net.predict(inputs_train);
    export_results("./train_predictions.csv", train_predictions);

    return 0;
}