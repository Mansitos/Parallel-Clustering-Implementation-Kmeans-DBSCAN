#ifndef DBSCANCUDA_H
#define DBSCANCUDA_H

#include <iostream>
#include <random>

void dbscan_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, std::mt19937 seed);

#endif