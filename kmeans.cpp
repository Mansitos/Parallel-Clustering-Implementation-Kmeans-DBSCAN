/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>
#include "utils.h"

int calculateCentroid(float* dataPoint, int dim, int k, float** centroids);
float euclideanDistance(float* dataPoint, float* centroid, int dim);
void updateCentroids(float** dataPoints, int length, int dim, float** centroids, int k);

void k_means(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, 10);

	// Local variables
	//const int k = 3;
	bool convergence = false;

	// 1. Choose the number of clusters(K) and obtain the data points
	if (k <= 0) {
		k = 1;
	}

	//2. Place the centroids c_1, c_2, .....c_k randomly
	float** centroids = new float*[k];
	for (int i = 0; i < k; i++) {
		centroids[i] = new float[dim];

		// rand init
		for (int j = 0; j < dim; j++) {
			centroids[i][j] = distrib(seed);	// random clusters
		}	
	}

	//3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
	while(!convergence) {

		bool convergenceCheck = true;

		//4. for each data point x_i:
		#pragma omp parallel for if(useParallelism)
		for (int i = 0; i < length; i++) {
			// find the nearest centroid(c_1, c_2 ..c_k)
			// assign the point to that cluster
			int newCentroid = calculateCentroid(dataPoints[i], dim, k, centroids);

			// When at least one centroid changed: convergence is not reached!
			if (dataPoints[i][dim] != newCentroid) {
				convergenceCheck = false;
				dataPoints[i][dim] = newCentroid;
			}
		}

		convergence = convergenceCheck;

		//5. for each cluster j = 1..k
		//- new centroid = mean of all points assigned to that cluster
		updateCentroids(dataPoints, length, dim, centroids, k);
	}

	//6. End (convergence reached)
	/*
	printf("Points:\n");
	printDataPoints(dataPoints, length, dim);
	printf("\n");
	printf("Final Centroids:\n");
	printCentroids(centroids, k, dim);
	*/
}

int calculateCentroid(float* dataPoint, int dim, int k, float** centroids) {
	int nearestCentroidIndex = dataPoint[dim];
	bool firstIteration = true;
	float minDistance = 0;
	for (int i = 0; i < k; i++) {
		float distance = euclideanDistance(dataPoint, centroids[i], dim);
		if (firstIteration) {
			firstIteration = false;
		}
		else if (distance < minDistance) {
			nearestCentroidIndex = i;
		}
		minDistance = distance;
	}
	return nearestCentroidIndex;
}

/*
Calculate the distance between two given points.
	@dataPoint: the first point
	@centroid: the second point; the centroid for which we want to know the distance
	@dim: dimension of the points

	Return: the distance (a float value)
*/
float euclideanDistance(float* dataPoint, float* centroid, int dim) {
	float sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += pow((dataPoint[i] - centroid[i]),2);
	}
	float distance = sqrt(sum);
	return distance;
}

/*
Update the centroid coordinates after a cycle of points-assignements-to-centroids
	@dataPoints: the list of points
	@length: number of points
	@dim: dimension of points (2D, 3D, etc.)
	@centroids: the list of centroids
	@k: number of centroids
*/
void updateCentroids(float** dataPoints, int length, int dim, float** centroids, int k) {
	for (int centroid = 0; centroid < k; centroid++) {

		for (int k = 0; k < dim; k++) {
			centroids[centroid][k] = 0;	// resetting actual centroid values
		}

		int assignedPoints = 0;	// counters to points assigned to actual centroid

		for (int point = 0; point < length; point++) {	// for each point...
			if (dataPoints[point][dim] == centroid) {	// if point was assigned to actual centroid

				for (int j = 0; j < dim; j++) {
					centroids[centroid][j] += dataPoints[point][j];
				}
			assignedPoints++;
			}
		}

		for (int j = 0; j < dim; j++) {
			if (assignedPoints != 0) {
				centroids[centroid][j] = centroids[centroid][j] / assignedPoints;
			} else {
				centroids[centroid][j] = 0;
			}
		}
	}
}