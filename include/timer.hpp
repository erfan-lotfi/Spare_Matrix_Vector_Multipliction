#pragma once
#include <chrono>

class Timer {
public:
    void start() {
        t1 = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t2 - t1).count();
    }

private:
    std::chrono::high_resolution_clock::time_point t1;
};

