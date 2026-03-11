#pragma once

#include "core.hpp"
#include <vector>
#include <cmath>

// Модуль математических операций
template <FloatingPoint T> // Проверяем, что это тип с плавающей точкой(core.hpp concept)
class MathOps {
public:
    // Вычисление линейной функции kx + b
    static T calculateLinear(T x, T k, T b) {
        return k * x + b;
    }


    // Матричное преобразование вектора (x, y) с помощью матрицы 2x2
    // Преобразуются координаты (x, y) типа T
    // Возвращает пару значений (newX, newY)

    static std::pair<T, T> transformVector(const Matrix<T>& mat, T x, T y) {
        
        // Умножение матрицы 2x2 на вектор (x, y)
        // Формула: newX = s11*x + s12*y; newY = s21*x + s22*y 
        T newX = mat(0, 0) * x + mat(0, 1) * y;
        T newY = mat(1, 0) * x + mat(1, 1) * y;
        
        return {newX, newY};
    }

    // Предсказание класса точки (x, y) относительно прямой y = kx + b
    static int predict(T x, T y, T k, T b) {
        T threshold = calculateLinear(x, k, b); // Вычисляем значение на прямой для данного x
        return (y > threshold) ? 1 : 0; // Если y выше прямой, класс 1, иначе 0
    }

    // Вычисление расстояния от точки (x, y) до прямой y = kx + b
    static T distanceToLine(T x, T y, T k, T b) {
        // Формула: |kx - y + b| / sqrt(k^2 + 1)
        return std::abs(k * x - y + b) / std::sqrt(k * k + 1);
    }
};
