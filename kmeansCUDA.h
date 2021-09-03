#ifndef KMEANSCUDA_H
#define KMEANSCUDA_H

#include <iostream>
#include <random>

void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed);

#endif