#pragma once

#include "core.hpp"
// Модуль математических операций
#include <cmath>
#include <utility>

namespace MathOps {
    // Проверяем, что это тип с плавающей точкой(core.hpp concept)
    template <FloatingPoint T>
    // Вычисление линейной функции kx + b
    inline T calculateLinear(T x, T k, T b) {
        return k * x + b;
    }

    // Матричное преобразование вектора (x, y) с помощью матрицы 2x2
    // Преобразуются координаты (x, y) типа T
    // Возвращает пару значений (newX, newY)
    template <FloatingPoint T>
    inline std::pair<T, T> transformVector(const Matrix<T>& mat, T x, T y) {
        
        // Умножение матрицы 2x2 на вектор (x, y)
        // Формула: newX = s11*x + s12*y; newY = s21*x + s22*y 
        T new_x = x * mat(0, 0) + y * mat(0, 1);
        T new_y = x * mat(1, 0) + y * mat(1, 1);
        return {new_x, new_y};
    }

    // Предсказание класса точки (x, y) относительно прямой y = kx + b
    template <FloatingPoint T>
    inline int predict(T x, T y, T k, T b) { // Вычисляем значение на прямой для данного x
    return (y > (k * x + b)) ? 1 : 0;
    }

    // Вычисление расстояния от точки (x, y) до прямой y = kx + b
    template <FloatingPoint T>
    inline T distanceToLine(T x, T y, T k, T b) {
        // Формула: |kx - y + b| / sqrt(k^2 + 1)
        return std::abs(k * x - y + b) / std::sqrt(k * k + 1);
    }
}