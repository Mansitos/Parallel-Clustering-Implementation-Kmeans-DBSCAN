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

/*
support function to clear the cluster column into dataPoints
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
*/
void clearClusterColumn(float** dataPoints, int length, int dim) {
	for (int i = 0; i < length; i++) {
		dataPoints[i][dim] = NULL;
	}
}

/*
support function to print the centroids
	@centroids: pointer to centroids
	@length: number of centroids
	@dim: dimension of centroids (2D, 3D etc.)
*/
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

/*
support function to print the dataPoints
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
*/
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

/*
support function to save dataPoints in a .csv file
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@filename: the name of the file
*/
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

/*
support function to convert an array of array in a single vector
	@output: vector of output
	@input: data input
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
*/
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

/*
support function to convert a single vector in an array of array
	@output: array of array of output
	@input: data input
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
*/
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

