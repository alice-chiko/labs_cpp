#include "include/core.hpp"
#include "include/math_ops.hpp"
#include "include/data_processor.hpp"
#include "include/metrics.hpp"
#include "include/logger.hpp"
#include "include/neural_net.hpp"
#include <fstream>


int main() {
    Logger::info("Lab #2: Learning with Noise and Split");

    float k = 0.8f;
    float b = 10.0f;
    
    // 1. Генерируем данные с шумом (noise_level = 5.0)
    auto full_dataset = DataProcessor::generateDataset(400, k, b, 5.0f);

    // 2. Нормализация всех данных 
    for (auto& pt : full_dataset) {
        pt.x /= 100.0f;
        pt.y /= 100.0f;
    }

    // 3. Разделяем на Train (80%) и Test (20%)
    auto [train_data, test_data] = DataProcessor::splitDataset(full_dataset, 0.8f);
    Logger::logValue("Train size", train_data.size());
    Logger::logValue("Test size", test_data.size());

    // 4. Обучаем перцептрон только на train_data
    NeuralNet::Perceptron<float> model(0.01f);
    model.train(train_data, 1000);

    // 5. Проверяем точность на test_data 
    int correct = 0;
    for (const auto& pt : test_data) {
        if (model.predict(pt.x, pt.y) == pt.label) {
            correct++;
        }
    }
    float accuracy = (float)correct / test_data.size() * 100.0f;
    Logger::logValue("Test Accuracy (%)", accuracy);

    // 6. Сохраняем для визуализации (возвращаем масштаб 100)
    for (auto& pt : full_dataset) {
        pt.x *= 100.0f;
        pt.y *= 100.0f;
    }
    DataProcessor::saveToCSV(full_dataset, "points.csv");

    float pred_k = model.get_k();
    float pred_b = model.get_b() * 100.0f;
    std::ofstream model_file("model_line.csv");
    model_file << "k,b\n" << pred_k << "," << pred_b << "\n";

    return 0;
}