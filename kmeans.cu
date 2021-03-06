/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>
#include "utils.h"
#include <omp.h>

int calculateCentroid(float* dataPoint, int dim, int k, float** centroids);
float euclideanDistance(float* dataPoint, float* centroid, int dim);
void updateCentroids(float** dataPoints, int length, int dim, float** centroids, int k,bool useParallelism);

/*
Main function call for kmeans execution.
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@useParallelism: to use or not OpenMP
	@seed: seed for the randomizer
*/
void k_means(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, length*100);

	// Local variables
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

	// Remove comments for centroids check
    // printCentroids(centroids, k, dim);

	//3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
	while(!convergence) {

		bool convergenceCheck = true;

		//4. for each data point x_i:
		#pragma omp parallel for schedule(static) if(useParallelism)
		for (int i = 0; i < length; i++) {
			// find the nearest centroid(c_1, c_2 ..c_k)
			// assign the point to that cluster
			int newCentroid = calculateCentroid(dataPoints[i], dim, k, centroids);
			// When at least one centroid changed: convergence is not reached!
			if ((int)(dataPoints[i][dim]) != newCentroid) {
				convergenceCheck = false;
				dataPoints[i][dim] = newCentroid;
			}
		}

		convergence = convergenceCheck;

		//5. for each cluster j = 1..k
		//- new centroid = mean of all points assigned to that cluster
		updateCentroids(dataPoints, length, dim, centroids, k,useParallelism);
	}
	//6. End (convergence reached)
}

/*
Assign the correct centroid to the dataPoint in base of the distance between centroid and dataPoint
	@dataPoint: pointer to actual point
	@dim: dimension of points (2D, 3D etc.)
	@k: number of centroids
	@centroids: pointer to the centroids

	Return: the nearest centroid to the dataPoint
*/
int calculateCentroid(float* dataPoint, int dim, int k, float** centroids) {
	int nearestCentroidIndex = dataPoint[dim];
	bool firstIteration = true;
	float minDistance = 0;
	for (int i = 0; i < k; i++) {
		float distance = euclideanDistance(dataPoint, centroids[i], dim);
		if (firstIteration) {
			firstIteration = false;
			nearestCentroidIndex = i;
			minDistance = distance;
		}
		else if (distance < minDistance) {
			nearestCentroidIndex = i;
			minDistance = distance;
		}
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
void updateCentroids(float** dataPoints, int length, int dim, float** centroids, int k,bool useParallelism) {
	#pragma omp parallel for schedule(dynamic) if(useParallelism)
	for (int centroid = 0; centroid < k; centroid++) {

		for (int j = 0; j < dim; j++) {
			centroids[centroid][j] = 0;	// resetting actual centroid values
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