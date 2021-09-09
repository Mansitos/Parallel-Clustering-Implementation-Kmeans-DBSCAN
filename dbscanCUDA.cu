#include "cuda_runtime.h"
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

/*

*/
__global__ void calculateDistancesCUDA(float* d_dataPoints, int core, int point, int dim, float* distance) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= dim)
		return;

	atomicAdd(&distance[0], pow(d_dataPoints[point+tid] - d_dataPoints[core+tid], 2));
	__syncthreads();

	if (tid == 0)
		distance[0] = sqrt(distance[0]);

	__syncthreads();
}

/*
Given a point index, returns an array containing the list of it's neighbours indexes
	@d_dataPoints:
	@neighbours:
	@index:
	@eps:
	@length:
	@dim:
	@count:
*/
__global__ void findNeighbours(float* d_dataPoints, float* neighbours, int index, float eps, int length, int dim, int* count) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	tid *= (dim + 1);
	if (tid >= length * (dim + 1))
		return;

	int TPB = 32*(length/32);	 // Threads x blocks
	if (length%32 != 0){
		TPB += 32;
	}

	// array of calculated distances for each point
	float* distance = (float*)malloc(sizeof(float));
	// initialization
	distance[0] = 0;

	calculateDistancesCUDA << <1, TPB >> >(d_dataPoints, index, tid, dim, distance);
	__syncthreads();

	if (threadIdx.x == 0)
		cudaDeviceSynchronize();
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
	@neighbours:
	@index:
	@neighCount:
*/
__global__ void neighboursDifference(float* neighbours, int index, int* neighCount) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= neighCount[0])
		return;
	if (neighbours[tid] == index) {
		index = 0;
		atomicAdd(&index, tid);
	}
	__syncthreads();

	for(int i=index+1; i<neighCount[0]; i++)
		neighbours[i - 1] = neighbours[i];
	__syncthreads();

	if (tid == 0)
		neighCount[0]--;
}

/*
...
	@neighbours: first array
	@neighboursChild: 2nd array
	@neighCount: length of 1st array
	@neighCountChild: length of 2nd array
	@index: array of positions of new neighbours
	@point: length of index
*/
__global__ void unionVectors(float* neighbours, float* neighboursChild, int* neighCount, int* neighCountChild, int* index, int* point) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= neighCountChild[0])
		return;
	int t = neighCount[0];
	bool check = false;
	for (int i = 0; i < t; i++) {
		if (neighbours[i] == neighboursChild[tid]) {
			check = true;
			break;
		}
	}

	if (!check) // if not found...
		index[atomicAdd(&point[0], 1)] = tid;
	__syncthreads();

	if (tid >= point[0])
		return;

	neighbours[atomicAdd(&neighCount[0], 1)] = neighboursChild[index[tid]];	
}

/*
....
	@dataPoints:
	@length:
	@dim:
	@clusterCounter:
	@minPts
	@eps:
*/
__global__ void dbscan_cuda_device(float* d_dataPoints, int length, int dim, float clusterCounter, float minPts, float eps) {
	// Points iteration. Step = dim of point + 1 (1 row)
	int step = dim+1;
	for (int i = 0; i < length*step; i += step) {
		// Check if actual point is not 0 (NULL - never assigned)
		if (d_dataPoints[i+dim] != 0) // 0 = NULL
			continue; // skip to next point

		int NumBlocks = length/256;
		if (length % 256 != 0) {
			NumBlocks += 1;
		}

		// array of neighbours allocation
		float* neighbours = (float*)malloc(sizeof(float) * length);
		// number of neighbours
		int* neighCount = (int*)malloc(sizeof(int));		// TODO: vedere se si può fare senza array
		neighCount[0] = 0;

		findNeighbours<<<NumBlocks,256>>>(d_dataPoints, neighbours, i, eps, length, dim, neighCount);
		cudaDeviceSynchronize();

		// if <minPts neighbours are found; assign NOISE to actual point and skip to next point
		if (neighCount[0]<minPts) {
			d_dataPoints[i+dim] = NOISE;
			continue;
		}

		// if >= minPts neighbours are found...
		// increase cluster counter and assign to actual point
		clusterCounter++;
		d_dataPoints[i+dim] = clusterCounter;

		int TPB = 32*(neighCount[0]/32);	 // Threads x blocks
		if (neighCount[0]%32 != 0){
			TPB += 32;
		}

		// remove actual point from linearized array
		neighboursDifference <<<1,TPB>>> (neighbours, i, neighCount);
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

			findNeighbours <<<NumBlocks, 256>>>(d_dataPoints, neighboursChild, neighbours[j], eps, length, dim, neighCountChild);
			cudaDeviceSynchronize();

			TPB = 32*(neighCountChild[0]/32);	 // Threads x blocks
			if (neighCountChild[0]%32 != 0){
				TPB += 32;
			}
			// if >= minPts neighbours are found...
			if (neighCountChild[0] >= minPts){
				int* index = (int*)malloc(sizeof(int)*(length-1));
				int* point = (int*)malloc(sizeof(int)); // TODO: change names.
				point[0] = 0;
				// Add found new neighbour's neighbours to neighbours list 
				unionVectors<<<1,TPB>>>(neighbours, neighboursChild, neighCount, neighCountChild, index, point);
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
	@dataPoints:
	@length:
	@dim:
	@useParallelism:
	@seed:
*/
void dbscan_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, sqrt(length) * 2);

	float clusterCounter = 0;
	const float minPts = 2;		// min number of points to create a new cluster
	float eps = distrib(seed);	// epsilon: min distance between 2 points to be considered neighbours

	// Linearization of dataPoints (from nD to 1D)
	float* h_dataPoints = new float[length * (dim + 1)];
	linealizer(h_dataPoints, dataPoints, length, dim + 1);

	float* d_dataPoints;

	int NumBlocks = length/256;
	if (length % 256 != 0) {
		NumBlocks += 1;
	}

	// device allocation for linearized array
	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(float) * length * (dim + 1)));
	// copy of host linearized array to device
	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyHostToDevice));

	// Main DBSCAN call
	dbscan_cuda_device << <1,1 >> > (d_dataPoints, length, dim, clusterCounter ,minPts,eps);

	// copy of device linearized array (result) to host
	HANDLE_ERROR(cudaMemcpy(h_dataPoints, d_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyDeviceToHost));

	// TODO REMOVE
	/*printf("DATAPOINTS\n");
	int counter = 1;
	for (int i = 0; i < length * (dim + 1); i++) {
		printf("%f ", h_dataPoints[i]);
		if (counter == (dim+1)) {
			printf("\n");
			counter = 0;
		}
		counter++;
	}*/

	delinealizer(dataPoints,h_dataPoints,length*(dim+1),dim);

	// Memory deallocation
	cudaFree(d_dataPoints);
	free(h_dataPoints);

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