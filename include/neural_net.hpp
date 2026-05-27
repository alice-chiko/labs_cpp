#pragma once
#include "core.hpp"
#include <vector>
#include <random>
#include <cmath>

namespace NeuralNet {
    template <FloatingPoint T>
    class Perceptron {
    private:
        std::vector<T> weights; // вектор весов динамического размера
        T bias;
        T learning_rate;

    public:
        Perceptron(size_t input_dim, T lr = 0.01) : learning_rate(lr) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-1.0, 1.0);
            
            weights.resize(input_dim);
            for (size_t i = 0; i < input_dim; ++i) {
                weights[i] = dist(gen);
            }
            bias = dist(gen);
        }

        int predict(const std::vector<T>& features) const {
            T z = bias;
            for (size_t i = 0; i < features.size(); ++i) {
                z += features[i] * weights[i];
            }
            return (z >= 0) ? 1 : 0;
        }

        void train(const std::vector<Point<T>>& dataset, int epochs) {
            for (int e = 0; e < epochs; ++e) {
                T epoch_error = 0;
                for (const auto& pt : dataset) {
                    int prediction = predict(pt.features);
                    T error = static_cast<T>(pt.label - prediction);
                    
                    if (error != 0) {
                        for (size_t i = 0; i < weights.size(); ++i) {
                            weights[i] += learning_rate * error * pt.features[i];
                        }
                        bias += learning_rate * error;
                        epoch_error += std::abs(error);
                    }
                }
                if (epoch_error == 0) break; 
            }
        }

        // геттеры для питона
        std::vector<T> get_weights() const { return weights; }
        T get_bias() const { return bias; }
    };
}