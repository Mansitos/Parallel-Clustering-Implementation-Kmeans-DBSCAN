#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include "utils.h"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line);

__device__ float euclideanDistance(float* dataPoint, float* centroid, int dim);

__device__ int calculateCentroid(float* dataPoint, int dim, int k, float** centroids);

bool convergence = false;

__device__ float euclideanDistance_device(float* dataPoint, float* centroid, int dim) {
	float sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += pow((dataPoint[i] - centroid[i]), 2);
	}
	float distance = sqrt(sum);
	return distance;
}

__device__ int calculateCentroid_device(float* dataPoint, int dim, int k, float** centroids) {
	int nearestCentroidIndex = dataPoint[dim];
	bool firstIteration = true;
	float minDistance = 0;
	for (int i = 0; i < k; i++) {
		float distance = euclideanDistance_device(dataPoint, centroids[i], dim);
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

__global__ void k_means_cuda_device(float* d_dataPoints, float* d_centroids, int length, int dim, int k) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < length) {
		printf("%f\n", d_dataPoints[tid]);
		//d_dataPoints[tid][dim] = calculateCentroid_device(d_dataPoints[tid], dim, k, d_centroids);
	}
}

void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, 10);

	float* h_dataPoints =new float[length*dim];
	float* h_centroids=new float[k*dim];

	// 1. Choose the number of clusters(K) and obtain the data points
	if (k <= 0) {
		k = 1;
	}

	//2. Place the centroids c_1, c_2, .....c_k randomly
	float** centroids = new float* [k];
	for (int i = 0; i < k; i++) {
		centroids[i] = new float[dim];

		// rand init
		for (int j = 0; j < dim; j++) {
			centroids[i][j] = distrib(seed);	// random clusters
		}
	}

	linealizer(h_dataPoints, dataPoints,length,dim);
	//linealizer(h_centroids, centroids, k, dim);

	float* d_dataPoints;
	float* d_centroids;

	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(float) * length * dim));
	/*HANDLE_ERROR(cudaMalloc((void**)&d_centroids, sizeof(float) * k * dim));

	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints,sizeof(float) * length * dim, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_centroids, h_centroids, sizeof(float) * k * dim, cudaMemcpyHostToDevice));

	int NumBlocks = length / 256;

	if (length % 256 != 0) {
		NumBlocks += 1;
	}

	k_means_cuda_device <<< NumBlocks, 256 >>> (d_dataPoints, d_centroids, length, dim, k);

	//for (int i = 0; i < length; i++)
		//	printf("%f\n", dataPoints[i][dim]);
		*/
	cudaFree(d_dataPoints);
	//cudaFree(d_centroids);
	delete h_dataPoints;
	delete h_centroids;

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