/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <random>

void k_means(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed);

#endif