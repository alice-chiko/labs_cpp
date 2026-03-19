#pragma once
#include "core.hpp"
#include "metrics.hpp" 
#include <vector>
#include <random>

namespace NeuralNet {
    template <FloatingPoint T>
    class Perceptron {
    private:
        Matrix<T> weights; // Матрица весов 1x2 (w_x, w_y) 
        T bias;
        T learning_rate;

    public:
        Perceptron(T lr = 0.01) : weights(1, 2), learning_rate(lr) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dist(-1.0, 1.0);
            
            weights(0, 0) = dist(gen); // w_x
            weights(0, 1) = dist(gen); // w_y
            bias = dist(gen);
        }

        int predict(T x, T y) const {
            // Используем матричное умножение 
            T z = x * weights(0, 0) + y * weights(0, 1) + bias;
            return (z >= 0) ? 1 : 0;
        }

        void train(const std::vector<Point<T>>& dataset, int epochs) {
            for (int e = 0; e < epochs; ++e) {
                T epoch_error = 0;
                for (const auto& pt : dataset) {
                    int prediction = predict(pt.x, pt.y);
                    T error = static_cast<T>(pt.label - prediction);
                    
                    if (error != 0) {
                        // Обновление весов через матрицу
                        weights(0, 0) += learning_rate * error * pt.x;
                        weights(0, 1) += learning_rate * error * pt.y;
                        bias += learning_rate * error;
                        epoch_error += std::abs(error);
                    }
                }
                // Если за всю эпоху не было ни одной ошибки — обучение закончено
                if (epoch_error == 0) break; 
            }
        }

        T get_k() const { return -weights(0, 0) / weights(0, 1); }
        T get_b() const { return -bias / weights(0, 1); }
    };
}