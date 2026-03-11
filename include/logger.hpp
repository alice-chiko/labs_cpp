#pragma once

#include <iostream>
#include <string>

class Logger {
public:
    static void info(const std::string& msg) {
        std::cout << "[INFO] " << msg << std::endl;
    }

    static void error(const std::string& msg) {
        std::cerr << "[ERROR] " << msg << std::endl;
    }

    template <typename T>
    static void logValue(const std::string& label, T value) {
        std::cout << "[DATA] " << label << ": " << value << std::endl;
    }

    static void testStatus(const std::string& testName, bool passed) {
        if (passed) {
            std::cout << "[TEST] " << testName << ": PASSED" << std::endl;
        } else {
            std::cerr << "[TEST] " << testName << ": FAILED" << std::endl;
        }
    }
};