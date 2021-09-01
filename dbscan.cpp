/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Christian Cagnoni & Mansi Andrea
*/

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <math.h>
#include <iterator>

#define NOISE -1

std::vector<float*> findNeighbours(float** dataPoints, float* dataPoint, float eps, int dim, int length);

std::vector<float*> difference(std::vector<float*> neighbours, float* dataPoint);

void unionVectors(std::vector<float*>* remainder, std::vector<float*> neighbours);

float calculateDistancesDB(float* dataPoints, float* dataPoint, int dim);

void dbscan(float** dataPoints, int length, int dim, bool useParallelism, std::mt19937 seed) {
	std::uniform_real_distribution<> distrib(0, sqrt(length) * 2);
	float count = 0;
	bool add = false;
	const float minPts = 2;
	float eps = distrib(seed);
	printf("Max distance threshold: %f\n", eps);
	for (int point = 0; point < length; point++) {
		float* dataPoint = dataPoints[point];
		if (dataPoint[dim] != 0)
			continue;
		std::vector<float*> neighbours = findNeighbours(dataPoints, dataPoint, eps, dim, length);

		if (neighbours.size() < minPts) {
			dataPoint[dim] = NOISE;
			continue;
		}
		
		count++;
		dataPoint[dim] = count;
		std::vector<float*> remainder = difference(neighbours, dataPoint);
		for (int i = 0; i < remainder.size(); i++) {
			float* dataPoint = remainder.at(i);
			if (dataPoint[dim] == NOISE)
				dataPoint[dim] = count;
			if (dataPoint[dim] != 0)
				continue;
			dataPoint[dim] = count;
			std::vector<float*> neighboursChild = findNeighbours(dataPoints, dataPoint, eps, dim, length);
			if (neighboursChild.size() >= minPts)
				unionVectors(&remainder, neighboursChild);
		}
	}
}

void unionVectors(std::vector<float*>* remainder, std::vector<float*> neighbours) {
	for (auto x : neighbours) {
		auto it = std::find((*remainder).begin(), (*remainder).end(), x);
		if (it == (*remainder).end()) {
			(*remainder).push_back(x);
		}
	}
}

std::vector<float*> difference(std::vector<float*> neighbours, float* dataPoint) {
	auto it = std::find(neighbours.begin(), neighbours.end(), dataPoint);

	// If element was found
	if (it != neighbours.end())
	{
		int index = it - neighbours.begin();
		neighbours.erase(neighbours.begin() + index);
	}
	return neighbours;
}

std::vector<float*> findNeighbours(float** dataPoints, float* dataPoint, float eps, int dim, int length) {
	std::vector<float*> neighbour;
	for (int i = 0; i < length; i++) {
		float* actualPoint = dataPoints[i];
		float distance = calculateDistancesDB(actualPoint, dataPoint, dim);
		if (distance <= eps) {
			neighbour.push_back(actualPoint);
		}
	}
	return neighbour;
}

float calculateDistancesDB(float* dataPoints, float* dataPoint, int dim) {
	float distance = 0;
	for (int i = 0; i < dim; i++)
		distance += pow(dataPoints[i] - dataPoint[i], 2);
	distance = sqrt(distance);
	return distance;
}