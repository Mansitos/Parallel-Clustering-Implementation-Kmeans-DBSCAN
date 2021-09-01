/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>

// Randomizer
const float minRange = 0;	// min value for coordinates
const float maxRange = 10;	// max value for coordinates
std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> distrib(minRange,maxRange);

static void initializeDataPoints(float** dataPoints, int length, int dim);
float** generateRandomInput(int numberOfPoints, int dimOfPoints);

/*
Generate a random instance for a clustering algorithm.
	@numberOfPoints: the amout of random points to be generated
	@dimOfPoints: the dimension of points to be generated
*/
float** generateRandomInput(int numberOfPoints, int dimOfPoints) {
	
	float** points = new float* [numberOfPoints];
	for (int i = 0; i < numberOfPoints; i++) {
		// The extra column is used for saving the assigned cluster for that point
		points[i] = new float[dimOfPoints + 1];	
	}

	initializeDataPoints(points, numberOfPoints, dimOfPoints);

	return points;

}

/*
Called by generateRandomInput: generates a random coordinate value for each dataPoint.
*/
void initializeDataPoints(float** dataPoints, int length, int dim) {
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < dim; j++) {
			dataPoints[i][j] = distrib(gen);
		}
		// Starting cluster is -1 ("null cluster")
		dataPoints[i][dim] = NULL;
	}
}