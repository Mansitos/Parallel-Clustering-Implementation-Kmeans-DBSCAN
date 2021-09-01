/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#ifndef UTILS_H
#define UTILS_H

void printCentroids(float** centroids, int length, int dim);
void printDataPoints(float** dataPoints, int length, int dim);
void saveToCsv(float** dataPoints, int length, int dim);
void clearClusterColumn(float** dataPoints, int length, int dim);

#endif
