#pragma once

#include "core.hpp"
#include <vector>
#include <cmath>
#include <numeric>

// Модуль метрик и функций потерь.
template <FloatingPoint T>
class Metrics {
public:
    /**
     * Реализация формулы Loss: сумма абсолютных отклонений.
     * Показывает суммарное расстояние от всех точек до разделяющей линии.
     */
    static T totalAbsoluteError(const std::vector<Point<T>>& points, T k, T b) {
        T sum = 0;
        for (const auto& p : points) {
            // отклонение y от линии kx+b
            sum += std::abs(p.y - (k * p.x + b));
        }
        return sum;
    }

    /**
     * Среднеквадратичная ошибка
     * Стандарт для нейросетей
     */
    static T mse(const std::vector<T>& predictions, const std::vector<int>& targets) {
        if (predictions.empty()) return 0;
        
        T errorSum = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            T diff = predictions[i] - static_cast<T>(targets[i]);
            errorSum += diff * diff;
        }
        return errorSum / static_cast<T>(predictions.size());
    }
};