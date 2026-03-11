#include "include/core.hpp"
#include "include/math_ops.hpp"
#include "include/data_processor.hpp"
#include "include/metrics.hpp"
#include "include/logger.hpp"

int main() {
    Logger::info("Laboratory Work #1");

    float k = 0.8f;
    float b = 10.0f;

    Logger::info("Generating dataset...");
    auto dataset = DataProcessor<float>::generateDataset(200, k, b);

    MatrixF transform(2, 2);
    transform(0,0) = 1.0f; transform(0,1) = 0.1f;
    transform(1,0) = 0.0f; transform(1,1) = 1.0f;

    auto [oldX, oldY] = std::make_pair(dataset[0].x, dataset[0].y);
    auto [newX, newY] = MathOps<float>::transformVector(transform, oldX, oldY);
    
    Logger::logValue("Original X", oldX);
    Logger::logValue("Transformed X", newX);

    float totalLoss = Metrics<float>::totalAbsoluteError(dataset, k, b);
    Logger::logValue("Final Absolute Loss", totalLoss);

    if (DataProcessor<float>::saveToCSV(dataset, "points.csv")) {
        Logger::info("Success: 'points.csv' generated for Python visualization.");
    }

    return 0;
}
