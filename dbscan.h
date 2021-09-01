#ifndef DBSCAN_H
#define DBSCAN_H

#include <iostream>
#include <random>

void dbscan(float** dataPoints, int length, int dim, bool useParallelism, std::mt19937 seed);

#endif
