#pragma once

#include "core.hpp"
#include <vector>
// Модуль метрик и функций потерь
#include <cmath>

namespace Metrics {
    // Сумма абсолютных ошибок для набора точек относительно линии y = kx + b
    // Чем меньше, тем лучше модель описывает данные
    // Формула: sum(|y - (kx + b)|) для всех точек
    template <FloatingPoint T>
    inline T totalAbsoluteError(const std::vector<Point<T>>& points, T k, T b) {
        T total_error = 0;
        for (const auto& p : points) {
            // отклонение y от линии kx+b
            T target_y = k * p.x + b;
            total_error += std::abs(p.y - target_y);
        }
        return total_error;
    }

    // Среднеквадратичная ошибка (MSE) для предсказаний модели
    // Формула: (1/n) * sum((pred_i - target_i)^2) для всех предсказаний и истинных классов
    // Чем меньше MSE, тем точнее модель предсказывает классы точек
    template <FloatingPoint T>
    inline T mse(const std::vector<T>& predictions, const std::vector<int>& targets) {
        if (predictions.empty()) return 0; // если вектор пустой - нет ошибок

        T sum_sq_error = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Вычисляем квадрат разницы между предсказанием и целью
            T diff = predictions[i] - static_cast<T>(targets[i]);
            sum_sq_error += diff * diff;
        }
        return sum_sq_error / static_cast<T>(predictions.size());
    }
}