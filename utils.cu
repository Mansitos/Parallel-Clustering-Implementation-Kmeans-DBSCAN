/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void clearClusterColumn(float** dataPoints, int length, int dim) {
	for (int i = 0; i < length; i++) {
		dataPoints[i][dim] = NULL;
	}
}

void printCentroids(float** centroids, int length, int dim) {
	for (int i = 0; i < length; i++) {
		printf("Centroid_%d: ", i);
		for (int j = 0; j < dim-1; j++) {
			printf("%f,", centroids[i][j]);
		}
		printf("%f", centroids[i][dim-1]);
		printf("%\n");
	}
}

void printDataPoints(float** dataPoints, int length, int dim) {
	for (int i = 0; i < length; i++) {
		printf("%d: ", i);
		printf(" (C: %d) ", (int)dataPoints[i][dim]);
		for (int j = 0; j < dim; j++) {
			printf("[%f]", dataPoints[i][j]);
		}
		printf("\n");
	}
}


void saveToCsv(float** dataPoints, int length, int dim, string filename) {
	ofstream file(filename);

	string firstline = "index;";
	for (int i = 0; i < dim; i++) {
		firstline.append(std::to_string(i));
		firstline.append(";");
	}
	firstline.append("cluster");
	firstline.append("\n");

	file << firstline;

	for (int i = 0; i < length; i++) {
		string line = (std::to_string(i + 1));
		line.append(";");

		for (int j = 0; j < dim; j++) {
			line.append(std::to_string(dataPoints[i][j]));
			line.append(";");
		}
		line.append(std::to_string((int)dataPoints[i][dim]));
		line.append("\n");

		file << line;
	}
}

void linealizer(float* output, float** input, int length, int dim) {
	int index = 0;
	for (int i = 0; i < length; i++)
		for (int j = 0; j < dim; j++)
			output[index++] = input[i][j];
}

void linealizer(double* output, float** input, int length, int dim) {
	int index = 0;
	for (int i = 0; i < length; i++)
		for (int j = 0; j < dim; j++)
			output[index++] = input[i][j];
}

void delinealizer(float** output, float* input, int length, int dim) {
	int index = 0;
	for (int i = dim; i < length; i += (dim+1))
		output[index++][dim] = input[i];
}

void delinealizer(float** output, double* input, int length, int dim) {
	int index = 0;
	for (int i = dim; i < length; i += (dim + 1))
		output[index++][dim] = input[i];
}

