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
#include <omp.h>

#define NOISE -1

std::vector<float*> findNeighbours(float** dataPoints, bool useParallelism, float* dataPoint, float eps, int dim, int length);

std::vector<float*> difference(std::vector<float*> neighbours, float* dataPoint);

void unionVectors(std::vector<float*>* remainder, bool useParallelism, std::vector<float*> neighbours);

float calculateDistancesDB(float* dataPoints, bool useParallelism, float* dataPoint, int dim);

float epsilonCalculation(float** dataPoints, int length, int dim, int minPts, bool useParallelism);

void dbscan(float** dataPoints, int length, int dim, bool useParallelism, std::mt19937 seed) {
	std::uniform_real_distribution<> distrib(0, (sqrt(length*10) * 2));
	float count = 0;
	int actualMinPts;
	const int defMinPts = 4;
	if (dim == 2) {
		actualMinPts = defMinPts;
	}
	else {
		int tmp = 2 * dim;
		while (tmp > length / 2) {
			if (tmp % 2 != 0) {
				tmp /= 2;
				tmp++;
			}
			else {
				tmp /= 2;
			}
		}
		actualMinPts = tmp;
	}
	const float minPts = actualMinPts;
	float eps = epsilonCalculation(dataPoints,length,dim,minPts,useParallelism);//sqrt(length * dim);//distrib(seed);
	printf("---->%d\n", length);
	for (int point = 0; point < length; point++) {
		float* dataPoint = dataPoints[point];
		if (dataPoint[dim] != 0)
			continue;
		std::vector<float*> neighbours = findNeighbours(dataPoints, useParallelism, dataPoint, eps, dim, length);
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
			std::vector<float*> neighboursChild = findNeighbours(dataPoints, useParallelism, dataPoint, eps, dim, length);
			if (neighboursChild.size() >= minPts)
				unionVectors(&remainder, useParallelism, neighboursChild);
		}
	}
	for (int i = 0; i < length; i++)
		printf("%f\n", dataPoints[i][dim]);
}

void unionVectors(std::vector<float*>* remainder, bool useParallelism, std::vector<float*> neighbours) {
	#pragma omp parallel for schedule(static) if(useParallelism)
	for (int i = 0; i < neighbours.size();i++) {
		bool find = false;
		for (int j = 0; j < (*remainder).size(); j++) {
			if ((*remainder).at(j) == neighbours.at(i)) {
				find = true;
			}
		}
		#pragma omp critical
		{
			if (!find) {
				(*remainder).push_back(neighbours[i]);
			}
		}
	}
}

float epsilonCalculation(float** dataPoints,int length,int dim,int minPts, bool useParallelism) {
	float* result = new float[length*minPts];
	float* dist = new float[minPts];
	int index = 0;
	int next = 0;
	for (int i = 0; i < minPts; i++)
		dist[i] = 0;
	for (int i = 0; i < length; i++) {
		float tmp = 0;
		for (int j = 0; j < length; j++) {
			tmp = calculateDistancesDB(dataPoints[i], useParallelism, dataPoints[j], dim);
			if (next < minPts) {
				dist[next++] = tmp;
			}
			else {
				for (int k = 0; k < minPts; k++) {
					if (tmp < dist[k]) {
						dist[k] = tmp;
						break;
					}
				}
			}
		}
		for (int j = 0; j < minPts; j++) {
			result[index++] = dist[j];
			dist[j] = 0;
		}
		next = 0;
	}
	float minDist = -1;
	for(int i=0;i<length*minPts;i++)
		for (int j = 0; j < length * minPts; j++) {
			if (minDist == -1) {
				float tmp = result[i] - result[j];
				if (tmp < 0)
					tmp *= (-1);
				if(tmp!=0)
					minDist = tmp;
			}
			else {
				float tmp= result[i] - result[j];
				if (tmp < 0)
					tmp *= (-1);
				if (tmp < minDist && tmp != 0)
					minDist = tmp;
			}
		}
	float eps = 0;
	int zero = 0;
	for (int i = 0; i < length * minPts; i++) {
		eps += result[i];
		if (result[i] == 0)
			zero++;
	}
	return (eps/((length*minPts)-zero)-minDist);
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

std::vector<float*> findNeighbours(float** dataPoints, bool useParallelism, float* dataPoint, float eps, int dim, int length) {
	std::vector<float*> neighbour;
	#pragma omp parallel for schedule(static) if(useParallelism)
	for (int i = 0; i < length; i++) {
		float* actualPoint = dataPoints[i];
		float distance = calculateDistancesDB(actualPoint, useParallelism, dataPoint, dim);
		#pragma omp critical
		{
			if (distance <= eps) {
				neighbour.push_back(actualPoint);
			}
		}
	}
	return neighbour;
}

float calculateDistancesDB(float* dataPoints, bool useParallelism, float* dataPoint, int dim) {
	float distance = 0;
	#pragma omp parallel for schedule(static) if(useParallelism)
	for (int i = 0; i < dim; i++)
		distance += pow(dataPoints[i] - dataPoint[i], 2);
	distance = sqrt(distance);
	return distance;
}