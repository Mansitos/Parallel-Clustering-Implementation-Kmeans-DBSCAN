/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>

static void initializeDataPoints(float** dataPoints, int length, int dim, std::mt19937 seed, std::string type);
float** generateRandomInput(int numberOfPoints, int dimOfPoints, std::mt19937 seed, std::string type);

/*
Generate a random instance for a clustering algorithm.
	@numberOfPoints: the amout of random points to be generated
	@dimOfPoints: the dimension of points to be generated
	@seed: seed for the randomizer
	@type: the type of randomizer, clusters or random

	Return: the pointer of the dataPoints
*/
float** generateRandomInput(int numberOfPoints, int dimOfPoints, std::mt19937 seed,std::string type) {
	
	float** points = new float* [numberOfPoints];
	for (int i = 0; i < numberOfPoints; i++) {
		// The extra column is used for saving the assigned cluster for that point
		points[i] = new float[dimOfPoints + 1];	
	}
	initializeDataPoints(points, numberOfPoints, dimOfPoints, seed,type);

	return points;
}

/*
Called by generateRandomInput: generates a random coordinate value for each dataPoint.
	@dataPoints: pointer to datapoints
	@lenght: the amout of random points to be initialized
	@dimOfPoints: the dimension of points to be initialized
	@seed: seed for the randomizer
	@type: the type of randomizer, clusters or random
*/
void initializeDataPoints(float** dataPoints, int length, int dim, std::mt19937 seed, std::string type) {
	// Randomizer
	const int minRange = 0;	// min value for coordinates
	const int maxRange = length*100;// max value for coordinates
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::uniform_real_distribution<> distrib(minRange, maxRange);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < dim; j++) {
			dataPoints[i][j] = distrib(seed);
		}
		// Starting cluster is -1 ("null cluster")
		dataPoints[i][dim] = NULL;
	}

	bool spreadPoints = false;

	if (type == "clusters") {
		spreadPoints = true;
	}
	else if (type != "random") {
		printf("Warning: only random and clusters are valid input random types! random will be used!\n");
	}

	int clusters = 5;
	if (spreadPoints) {
		int step = (length / clusters);
		int mulfactor = length / 20;
		//printf("step:%d\n", step);
			for (int c = 1; c <= clusters; c++) {
				for (int k = step*(c-1); k < step*c; k++) {
					dataPoints[k][0] = step * 100 * (c - 1);
					//dataPoints[k][0] += mulfactor *step*(c-1);
			}
		}
	}
}