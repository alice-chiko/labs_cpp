#pragma once
#include "core.hpp"
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>

namespace DataProcessor {
    // Генерация данных с шумом
    template <FloatingPoint T>
    inline std::vector<Point<T>> generateDataset(size_t count, T k, T b, T noise_level = 0.0) {
        std::vector<Point<T>> dataset;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0.0, 100.0);

        for (size_t i = 0; i < count; ++i) {
            T x = dist(gen);
            T y = dist(gen);
            int label = (y > (k * x + b)) ? 1 : 0;

            // Добавляем шум только если он больше нуля
            if (noise_level > 0) {
                std::normal_distribution<T> noise(0.0, noise_level);
                x += noise(gen);
                y += noise(gen);
            }

            dataset.emplace_back(x, y, label);
        }
        return dataset;
    }

    // Разделение на обучающую и тестовую выборки
    template <typename T>
    inline std::pair<std::vector<T>, std::vector<T>> splitDataset(std::vector<T> data, float train_ratio = 0.8f) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);

        size_t train_size = static_cast<size_t>(data.size() * train_ratio);
        std::vector<T> train_data(data.begin(), data.begin() + train_size);
        std::vector<T> test_data(data.begin() + train_size, data.end());

        return {train_data, test_data};
    }

    // Сохранение в CSV 
    template <FloatingPoint T>
    inline bool saveToCSV(const std::vector<Point<T>>& data, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "x,y,label\n";
        for (const auto& p : data) {
            file << p.x << "," << p.y << "," << p.label << "\n";
        }
        file.close();
        return true;
    }
}