#pragma once // Защита от повторного включения файла

#include <vector>
#include <concepts>
#include <type_traits>

// Используем концепты для проверки типов данных
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Структура точки для бинарного классификатора. 
template <FloatingPoint T>
struct Point {
    T x;           // Координата X
    T y;           // Координата Y
    int label;     // Класс (0 или 1) 
    Point(T x_val, T y_val, int l = 0) : x(x_val), y(y_val), label(l) {}
};

// Матрица
// Обертка над std::vector для хранения весов или данных
template <FloatingPoint T>
struct Matrix {
    std::vector<T> data;
    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

    // Оператор доступа к элементам 
    T& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const T& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }
};

// Создание псевдонимов типов
using PointF = Point<float>;
using MatrixF = Matrix<float>;