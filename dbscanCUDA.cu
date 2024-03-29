﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include "utils.h"
#include <math.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define NOISE -1

static void HandleError(cudaError_t err, const char* file, int line);

float calculateDistancesDB_CUDA(float* dataPoints, float* dataPoint, int dim);

float epsilonCalculation_CUDA(float** dataPoints, int length, int dim, int minPts);

__device__ unsigned int countBD = 0;	//counter for CUDA blocks
__device__ int threadXblock = 1024;		//number of thread per block
__device__ int lockD = 0;				//variable to lock access to part of code

/*
Calculates the distance between 2 points.
	@d_dataPoints: device pointer to all points
	@core: index of first point
	@point: index of second point
	@dim: dimension of points
	@distance: result pointer 
*/
__global__ void calculateDistancesCUDA(float* d_dataPoints, int core, int point, int dim, float* distance) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= dim)
		return;
	atomicAdd(&distance[0], pow(d_dataPoints[point+tid] - d_dataPoints[core+tid], 2));
}

/*
Given a point index, returns an array containing the list of it's neighbours indexes and the number (count)
	@d_dataPoints: device pointer to all points
	@neighbours: result pointer
	@index: the index of the point of which neighbours must be computed
	@eps: epsilon threshold (distance < eps then point is neighbour)
	@length: number of points
	@dim: dimension of points
	@count: the number of found neighbours
*/
__global__ void findNeighbours(float* d_dataPoints, float* neighbours, int index, float eps, int length, int dim, int* count) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	tid *= (dim + 1);
	if (tid >= length * (dim + 1))
		return;

	int NumBlocks = dim / threadXblock;
	if (dim % threadXblock != 0)
		NumBlocks++;

	// array of calculated distances for each point
	float* distance = (float*)malloc(sizeof(float));
	// initialization
	distance[0] = 0;

	calculateDistancesCUDA <<<NumBlocks, threadXblock >>>(d_dataPoints, index, tid, dim, distance);
	__syncthreads();

	if (threadIdx.x == 0)
		cudaDeviceSynchronize();
	__syncthreads();

	distance[0] = sqrt(distance[0]);

	__syncthreads();


	if (distance[0] <= eps) {
		int t = atomicAdd(&count[0], 1);
		neighbours[t]=tid;
	}
	__syncthreads();
	free(distance);
}
/*
Remove actual point from linearized array
	@neighbours: device pointer to the list of neighbours (indexes) from which the point must be removed
	@index: index of the point to be removed
	@neighCount: the number of total neighbours
*/
__global__ void neighboursDifference(float* neighbours, int index, int* neighCount,int NumBlocks) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= neighCount[0])
		return;

	if ((int)(neighbours[tid]) == index) {
		index = 0;
		atomicAdd(&index, tid);
	}
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(&countBD, 1);
	__syncthreads();

	if (countBD >= NumBlocks && atomicAdd(&lockD, 1) < 1) {
		countBD = 0;
		for (int i = index + 1; i < neighCount[0]; i++)
			neighbours[i - 1] = neighbours[i];
		neighCount[0]--;
		lockD = 0;
	}
}

/*
Support function of unionVectors()
	@neighbours: device pointer to the list of neighbours
	@neighboursChild:
	@neighCount: pointer to neighbours counters
	@index: 
	@point: 
*/
__global__ void unionVectorsSupport(float* neighbours, float* neighboursChild, int* neighCount, int* index, int* point) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= point[0])
		return;
	neighbours[atomicAdd(&neighCount[0], 1)] = neighboursChild[index[tid]];
}

/*
Given neighbours list of the previous iteration, it adds/updates the neighbours points list (adding their indexes) of new neighbours
	@neighbours: first array
	@neighboursChild: 2nd array
	@neighCount: length of 1st array
	@neighCountChild: length of 2nd array
	@index: array of positions of new neighbours
	@point: length of index
*/
__global__ void unionVectors(float* neighbours, float* neighboursChild, int* neighCount, int* neighCountChild, int* index, int* point, int NumBlocks) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= neighCountChild[0])
		return;
	int t = neighCount[0];
	bool check = false;
	for (int i = 0; i < t; i++) {
		if ((int)(neighbours[i]) == (int)(neighboursChild[tid])) {
			check = true;
			break;
		}
	}

	if (threadIdx.x == 0)
		atomicAdd(&countBD, 1);

	if (!check) // if not found...
		index[atomicAdd(&point[0], 1)] = tid;
	__syncthreads();

	if (countBD >= NumBlocks && atomicAdd(&lockD, 1) < 1) {
		countBD = 0;
		int NumBlocksSupport = point[0] / threadXblock;
		if (point[0] % threadXblock != 0)
			NumBlocksSupport++;
		unionVectorsSupport << <NumBlocksSupport, threadXblock >> > (neighbours, neighboursChild, neighCount, index, point);
		cudaDeviceSynchronize();
		lockD = 0;
	}
}

/*
Main function call for dbscan cuda execution.
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@clusterCounter: counter of clusters
	@minPts: minimum number of points to be considered a new cluster
	@eps: min. distance for a point to be considered inside a given cluster/neighbours of another point
*/
__global__ void dbscan_cuda_device(float* d_dataPoints, int length, int dim, float clusterCounter, float minPts, float eps) {
	if (threadIdx.x > 0)
		return;

	// Points iteration. Step = dim of point + 1 (1 row)
	int step = dim+1;
	for (int i = 0; i < length*step; i += step) {
		// Check if actual point is not 0 (NULL - never assigned)
		if (d_dataPoints[i+dim] != 0) // 0 = NULL
			continue; // skip to next point

		int NumBlocks = length/ threadXblock;
		if (length % threadXblock != 0) {
			NumBlocks += 1;
		}

		// array of neighbours allocation
		float* neighbours = (float*)malloc(sizeof(float) * length);
		// number of neighbours
		int* neighCount = (int*)malloc(sizeof(int));		// TODO: vedere se si può fare senza array
		neighCount[0] = 0;

		findNeighbours<<<NumBlocks, threadXblock >>>(d_dataPoints, neighbours, i, eps, length, dim, neighCount);
		cudaDeviceSynchronize();

		// if <minPts neighbours are found; assign NOISE to actual point and skip to next point
		if (neighCount[0]<minPts) {
			d_dataPoints[i+dim] = NOISE;
			free(neighbours);
			free(neighCount);
			continue;
		}

		// if >= minPts neighbours are found...
		// increase cluster counter and assign to actual point
		clusterCounter++;
		d_dataPoints[i+dim] = clusterCounter;

		NumBlocks = neighCount[0] / threadXblock;
		if (neighCount[0] % threadXblock != 0) {
			NumBlocks += 1;
		}

		// remove actual point from linearized array
		neighboursDifference <<<NumBlocks, threadXblock >>> (neighbours, i, neighCount,NumBlocks);
		cudaDeviceSynchronize();

		// iterate through neighbours
		for (int j = 0; j < neighCount[0]; j++) {
			if (d_dataPoints[(int)neighbours[j] + dim] == NOISE) // it was noise before, now is assigned to actual point's cluster (TODO REMOVE)
				d_dataPoints[(int)neighbours[j] + dim] = clusterCounter;
			if (d_dataPoints[(int)neighbours[j] + dim] != 0)	 // it was already assigned: not iteration needed
				continue;
			d_dataPoints[(int)neighbours[j] + dim] = clusterCounter;	// assign to actual point's cluster

			// array of neighbours allocation (for the actual neighbour)
			float* neighboursChild = (float*)malloc(sizeof(float) * length);
			int* neighCountChild = (int*)malloc(sizeof(int));
			neighCountChild[0] = 0;

			NumBlocks = length / threadXblock;
			if (length % threadXblock != 0) {
				NumBlocks += 1;
			}

			findNeighbours <<<NumBlocks, threadXblock >>>(d_dataPoints, neighboursChild, neighbours[j], eps, length, dim, neighCountChild);
			cudaDeviceSynchronize();

			NumBlocks = neighCountChild[0] / threadXblock;
			if (neighCountChild[0] % threadXblock != 0)
				NumBlocks++;

			// if >= minPts neighbours are found...
			if (neighCountChild[0] >= minPts){
				int* index = (int*)malloc(sizeof(int)*length);
				int* point = (int*)malloc(sizeof(int)); // TODO: change names.
				point[0] = 0;
				// Add found new neighbour's neighbours to neighbours list 
				unionVectors<<<NumBlocks, threadXblock >>>(neighbours, neighboursChild, neighCount, neighCountChild, index, point,NumBlocks);
				cudaDeviceSynchronize();
				free(index);
				free(point);
			}
			free(neighboursChild);
			free(neighCountChild);
		}
		free(neighbours);
		free(neighCount);
	}
}



/*
Main DBSCAN call. Host initialization.
	@dataPoints: point to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@seed: the seed for the input generator
*/
void dbscan_cuda_host(float** dataPoints, int length, int dim, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, (sqrt(length*10) * 2));

	float clusterCounter = 0;
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
	const int minPts = actualMinPts;		// min number of points to create a new cluster
	float eps = epsilonCalculation_CUDA(dataPoints,length,dim,minPts); // epsilon: min distance between 2 points to be considered neighbours

	// Linearization of dataPoints (from nD to 1D)
	float* h_dataPoints = new float[length * (dim + 1)];
	linealizer(h_dataPoints, dataPoints, length, dim + 1);

	float* d_dataPoints;

	// device allocation for linearized array
	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(float) * length * (dim + 1)));
	// copy of host linearized array to device
	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyHostToDevice));

	// Main DBSCAN call
	dbscan_cuda_device <<<1,32>>> (d_dataPoints, length, dim, clusterCounter, minPts, eps);

	// copy of device linearized array (result) to host
	HANDLE_ERROR(cudaMemcpy(h_dataPoints, d_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyDeviceToHost));

	delinealizer(dataPoints,h_dataPoints,length*(dim+1),dim);

	// Memory deallocation
	cudaFree(d_dataPoints);
	free(h_dataPoints);

}

/*
Calculates an appropriate epsilon value for the given randomized input. (Host side - serial)
	@dataPoints: pointer to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@minPts: threshold of neighbours

	Return: the epsilon parameter for the function to work
*/
float epsilonCalculation_CUDA(float** dataPoints, int length, int dim, int minPts) {
	float* result = new float[length * minPts];
	float* dist = new float[minPts];
	int index = 0;
	int next = 0;
	//reset dist array
	for (int i = 0; i < minPts; i++)
		dist[i] = 0;
	//get the minPts points that have minimum distance for all points
	for (int i = 0; i < length; i++) {
		float tmp = 0;
		for (int j = 0; j < length; j++) {
			tmp = calculateDistancesDB_CUDA(dataPoints[i], dataPoints[j], dim);
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
	//get the minimum distance between the all minPts
	float minDist = -1;
	for (int i = 0; i < length * minPts; i++)
		for (int j = 0; j < length * minPts; j++) {
			if (minDist == -1) {
				float tmp = result[i] - result[j];
				if (tmp < 0)
					tmp *= (-1);
				if (tmp != 0)
					minDist = tmp;
			}
			else {
				float tmp = result[i] - result[j];
				if (tmp < 0)
					tmp *= (-1);
				if (tmp < minDist && tmp != 0)
					minDist = tmp;
			}
		}
	//count the elements that have distance equal to zero
	//and add the distances to eps
	float eps = 0;
	int zero = 0;
	for (int i = 0; i < length * minPts; i++) {
		eps += result[i];
		if (result[i] == 0)
			zero++;
	}
	//return the mean of eps adjust with count of elements equal to zero and minimum distance
	return (eps / ((length * minPts) - zero) - minDist);
}

/*
Calculates the distance between 2 points.
	@dataPoints: pointer to all points
	@dataPoint: the actual point
	@dim: dimension of points

	Return: the distance between dataPoints and dataPoint
*/
float calculateDistancesDB_CUDA(float* dataPoints, float* dataPoint, int dim) {
	float distance = 0;
	for (int i = 0; i < dim; i++)
		distance += pow(dataPoints[i] - dataPoint[i], 2);
	distance = sqrt(distance);
	return distance;
}

/*
Handles CUDA Errors and print them.
*/
static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}