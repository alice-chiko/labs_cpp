#include "include/math_ops.hpp"
#include "include/metrics.hpp"
#include "include/data_processor.hpp"
#include "include/logger.hpp"
#include "include/neural_net.hpp" 
#include <cassert>
#include <vector>
void test_math() {
    float res = MathOps::calculateLinear(2.0f, 3.0f, 1.0f);
    Logger::testStatus("Math: Linear y=kx+b", res == 7.0f);

    MatrixF mat(2, 2);
    mat(0,0)=1; mat(0,1)=2; mat(1,0)=3; mat(1,1)=4;
    auto [nx, ny] = MathOps::transformVector(mat, 1.0f, 1.0f);
    Logger::testStatus("Math: Matrix Transform", (nx == 3.0f && ny == 7.0f));
}

void test_metrics() {
    std::vector<PointF> pts = { PointF(1.0f, 10.0f, 1) };
    float loss = Metrics::totalAbsoluteError(pts, 1.0f, 5.0f);
    Logger::testStatus("Metrics: Absolute Loss", loss == 4.0f);
}

void test_data() {
    auto dataset = DataProcessor::generateDataset(50, 1.0f, 5.0f);
    Logger::testStatus("Data: Dataset size", dataset.size() == 50);
    
    bool valid = (dataset[0].y > (1.0f * dataset[0].x + 5.0f)) ? (dataset[0].label == 1) : (dataset[0].label == 0);
    Logger::testStatus("Data: Label correctness", valid);
}

void test_perceptron_learning() {
    NeuralNet::Perceptron<float> nn(0.1f);
    
    // Создаем простейший набор: точка (0,0) - класс 0, точка (1,1) - класс 1
    std::vector<PointF> small_data = { PointF(0.0f, 0.0f, 0), PointF(1.0f, 1.0f, 1) };
    
    nn.train(small_data, 100);
    
    bool works = (nn.predict(0.0f, 0.0f) == 0 && nn.predict(1.0f, 1.0f) == 1);
    Logger::testStatus("NeuralNet: Basic Convergence", works);
}

int main() {
    Logger::info("Running Unit Tests...");
    try {
        test_math();
        test_metrics();
        test_data();
        test_perceptron_learning(); 
        Logger::info("All tests passed successfully!");
    } catch (...) {
        Logger::error("Tests failed with an exception.");
        return 1;
    }
    return 0;
}