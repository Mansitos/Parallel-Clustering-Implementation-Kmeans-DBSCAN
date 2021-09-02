#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line);

bool convergence = false;

__global__ void k_means_cuda_device(float** d_dataPoints,float**) {

}

void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, 10);

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

	float** d_dataPoints;
	float** d_centroids;
	int d_length;
	int d_dim;
	int d_k;

	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, length*sizeof(float*)));
	HANDLE_ERROR(cudaMalloc((void**)&d_centroids, k * sizeof(float*)));

	HANDLE_ERROR(cudaMemcpy(d_dataPoints, dataPoints, length * sizeof(float*), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_centroids, centroids, length * sizeof(float*), cudaMemcpyHostToDevice));

	int NumBlocks = length / 256;

	if (length % 256 != 0) {
		NumBlocks += 1;
	}



	cudaFree(d_dataPoints);
	cudaFree(d_centroids);

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