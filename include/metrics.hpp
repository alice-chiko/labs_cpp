#pragma once

#include "core.hpp"
#include <vector>
#include <cmath>

// Модуль метрик и функций потерь
template <FloatingPoint T>
class Metrics {
public:
    // Сумма абсолютных ошибок для набора точек относительно линии y = kx + b
    // Чем меньше, тем лучше модель описывает данные
    // Формула: sum(|y - (kx + b)|) для всех точек
    static T totalAbsoluteError(const std::vector<Point<T>>& points, T k, T b) {
        T sum = 0;
        for (const auto& p : points) {
            // отклонение y от линии kx+b
            sum += std::abs(p.y - (k * p.x + b));
        }
        return sum;
    }

    // Среднеквадратичная ошибка (MSE) для предсказаний модели
    static T mse(const std::vector<T>& predictions, const std::vector<int>& targets) {
        if (predictions.empty()) return 0; // если вектор пустой - нет ошибок
        
        T errorSum = 0;
        for (size_t i = 0; i < predictions.size(); ++i) 
        // Вычисляем квадрат разницы между предсказанием и истинным классом для каждой точки
        {
            T diff = predictions[i] - static_cast<T>(targets[i]); // Приводим целочисленный класс к типу T для корректного вычитания
            errorSum += diff * diff; 
        }
        return errorSum / static_cast<T>(predictions.size());
    }
};
