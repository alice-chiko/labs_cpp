#include "include/math_ops.hpp"
#include "include/metrics.hpp"
#include "include/data_processor.hpp"
#include "include/logger.hpp"
#include <cassert>
#include <vector>

void test_math() {
    // 1. Тест линейной функции y = kx + b
    float res = MathOps<float>::calculateLinear(2.0f, 3.0f, 1.0f); // 3*2 + 1 = 7
    Logger::testStatus("Math: Linear y=kx+b", res == 7.0f);

    // 2. Тест матричного умножения
    MatrixF mat(2, 2);
    mat(0,0)=1; mat(0,1)=2; mat(1,0)=3; mat(1,1)=4;
    auto [nx, ny] = MathOps<float>::transformVector(mat, 1.0f, 1.0f); // x=1+2=3, y=3+4=7
    Logger::testStatus("Math: Matrix Transform", (nx == 3.0f && ny == 7.0f));
}

void test_metrics() {
    // Тест Loss (сумма абсолютных отклонений)
    std::vector<PointF> pts = { PointF(1.0f, 10.0f, 1) }; // x=1, y=10
    // Если линия y = 1*x + 5, то ожидаемый y_line = 6. Отклонение |10 - 6| = 4
    float loss = Metrics<float>::totalAbsoluteError(pts, 1.0f, 5.0f);
    Logger::testStatus("Metrics: Absolute Loss", loss == 4.0f);
}

void test_data() {
    // Тест генератора данных
    auto dataset = DataProcessor<float>::generateDataset(50, 1.0f, 5.0f);
    Logger::testStatus("Data: Dataset size", dataset.size() == 50);
    
    // Проверка, что метка соответствует положению относительно линии
    bool valid = (dataset[0].y > (1.0f * dataset[0].x + 5.0f)) ? (dataset[0].label == 1) : (dataset[0].label == 0);
    Logger::testStatus("Data: Label correctness", valid);
}

int main() {
    Logger::info("Running Unit Tests...");
    try {
        test_math();
        test_metrics();
        test_data();
        Logger::info("All tests passed successfully!");
    } catch (...) {
        Logger::error("Tests failed with an exception.");
        return 1;
    }
    return 0;
}