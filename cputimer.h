#pragma once
/**
 *
 */

#include <time.h>
#include <chrono>

struct CpuTimer
{

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;


    void Start()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void Stop()
    {
        stop = std::chrono::high_resolution_clock::now();
    }

    std::chrono::duration<double> Elapsed()
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    }
};