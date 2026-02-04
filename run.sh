#!/usr/bin/env bash

echo "Running project..."
g++ -std=c++20 -O3 -fopenmp src/*.cpp -I src -o network
nice -n +19 ./network