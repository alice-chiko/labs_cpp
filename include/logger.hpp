#pragma once
#include <iostream>
#include <string>

namespace Logger {
    inline void info(const std::string& msg) {
        std::cout << "[INFO] " << msg << std::endl;
    }

    inline void error(const std::string& msg) {
        std::cerr << "[ERROR] " << msg << std::endl;
    }

    template <typename T>
    inline void logValue(const std::string& label, T value) {
        std::cout << "[DATA] " << label << ": " << value << std::endl;
    }

    inline void testStatus(const std::string& testName, bool passed) {
        if (passed) {
            std::cout << "[TEST] " << testName << ": PASSED" << std::endl;
        } else {
            std::cerr << "[TEST] " << testName << ": FAILED" << std::endl;
        }
    }
}