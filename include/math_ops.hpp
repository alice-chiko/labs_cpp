#pragma once

#include "core.hpp"
#include <vector>
#include <cmath>

// Модуль математических операций.
template <FloatingPoint T>
class MathOps {
public:
    /**
     * 1. Уравнение прямой
     * Вычисляет значение y для заданного x: y = k*x + b
     * Используется для определения разделяющей линии
     */
    static T calculateLinear(T x, T k, T b) {
        return k * x + b;
    }

    /**
     * 2. Матричное умножение 2x2 на вектор (x, y)
     */
    static std::pair<T, T> transformVector(const Matrix<T>& mat, T x, T y) {
        // Проверка: матрица должна быть как минимум 2x2 
        if (mat.data.size() < 4) return {x, y};

        // Формула: x_new = s11*x + s12*y; y_new = s21*x + s22*y 
        T newX = mat(0, 0) * x + mat(0, 1) * y;
        T newY = mat(1, 0) * x + mat(1, 1) * y;
        
        return {newX, newY};
    }

    /**
     * 3. Определение знака 
     * Помогает понять, с какой стороны от линии находится точка.
     * Если y > kx + b, результат положительный.
     */
    static int predict(T x, T y, T k, T b) {
        T threshold = calculateLinear(x, k, b);
        return (y > threshold) ? 1 : 0;
    }

    /**
     * 4. Расстояние от точки до прямой (перпендикуляр)
     */
    static T distanceToLine(T x, T y, T k, T b) {
        // Формула: |kx - y + b| / sqrt(k^2 + 1)
        return std::abs(k * x - y + b) / std::sqrt(k * k + 1);
    }
};