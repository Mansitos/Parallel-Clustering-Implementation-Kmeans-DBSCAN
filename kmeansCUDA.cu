#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include "utils.h"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line);
__device__ float euclideanDistance(float* dataPoint, float* centroid, int dim);
__device__ int calculateCentroid(float* dataPoint, int dim, int k, float** centroids);

__device__ unsigned int countB = 0;			// counter for CUDA blocks
__device__ unsigned int lock = 0;			// variable to lock access to part of code
__device__ int convergenceThreshold = 0; 	// 0%
__device__ int errorsCounter = 0; 			// the amount of different results
__device__ int assigned = 0;				// counter of elements assigned 

/*
Atomic add implementation for double
	@address: the address of the value to be updatig
	@val: the value to be adding

	Return: the old value of address
*/
__device__ double atomicAddDouble(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

/*
Function for euclidean distance calculation between 2 given points
	@dataPoint: the list of points (1st points)
	@tid: the index of the first point
	@centroid: the list of centroids (2nd points)
	@index: the index of the second points
	@dim: dimension of points: 3D, 4D, etc.

	Return: the distance between dataPoint and centroid
*/
__device__ double euclideanDistance_device(double* dataPoint, int tid, double* centroid, int index, int dim) {
	double sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += pow((dataPoint[tid + i] - centroid[index + i]), 2);
	}
	double distance = sqrt(sum);
	return distance;
}

/*
Function for assigning a centroid to a given point. Iterates each centroid and check which one is the closest one.
	@dataPoint: the list of points
	@tid: the index of the point for which centroid have to be calculated
	@dim: dimension of points: 3D etc.
	@k: number of centroids
	@centroids: list of centroids.

	Return: the index of the nearest centroid 
*/
__device__ int calculateCentroid_device(double* dataPoint, int tid, int dim, int k, double* centroids) {
	int nearestCentroidIndex = dataPoint[tid + dim];
	bool firstIteration = true;
	double minDistance = 0;
	for (int i = 0; i < k; i++) {
		double distance = euclideanDistance_device(dataPoint, tid, centroids, i * dim, dim);
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
CUDA Kernel for parallel computing of updated centroids coordinates. (parallel division)
	@d_centroids: the list of centroids (device side)
	@assignedPoints: the amount of assigned points to that cluster
	@dim: dimension of points
	@index: the index of the cluster to update
*/
__global__ void computeCentroids(double* d_centroids, int assignedPoint, int dim, int index) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dim)
		return;
	d_centroids[index + tid] /= assignedPoint;
}

/*
CUDA Kernel for centroids update.
	@d_dataPoints: the list of points (device side)
	@d_centroids: the list of centroids (device side)
	@assignedPoints: the amount of assigned points to that cluster
	@dim: dimension of points
	@lenght: amount of points
	@k: number of centroids
	@NumBlocks: the number of blocks which are running this kernel
*/
__global__ void k_means_cuda_device_update_centroids(double* d_dataPoints, double* d_centroids, int* assignedPoints, int length, int dim, int k, int NumBlocks) {
	int threadsXblock = 1024;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length * (dim + 1))
		return;

	if (threadIdx.x == 0)
		atomicAdd(&countB, 1);
	__syncthreads();

	extern __shared__ double s[];
	double* b_centroids = s;
	int* b_assignedPoints = (int*)&b_centroids[k * (dim + 1)];
	if (threadIdx.x < k)
		b_assignedPoints[threadIdx.x] = 0;
	int mul = 0;
	while (threadIdx.x + mul * threadsXblock < k * dim) {
		b_centroids[threadIdx.x + mul * threadsXblock] = 0;
		mul++;
	}
	__syncthreads();

	// Sum of point's coordinates for each cluster
	int clusterIndex = ((int)(tid / (dim + 1))) * (dim + 1) + dim;
	int coordOffest = tid % (dim + 1);

	if (tid != clusterIndex) {
		atomicAddDouble(&b_centroids[(int)d_dataPoints[clusterIndex] * dim + coordOffest], d_dataPoints[tid]);
	}
	else {
		atomicAdd(&b_assignedPoints[(int)d_dataPoints[tid]], 1);
	}

	__syncthreads();

	if (threadIdx.x < k)
		atomicAdd(&assignedPoints[threadIdx.x], b_assignedPoints[threadIdx.x]);
	mul = 0;
	while (threadIdx.x + mul * threadsXblock < k * dim) {
		atomicAddDouble(&d_centroids[threadIdx.x + mul * threadsXblock], b_centroids[threadIdx.x + mul * threadsXblock]);
		mul++;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(&countB, 1);
	}

	__syncthreads();

	if (countB >= 2 * NumBlocks) {
		if (threadIdx.x < k && atomicAdd(&lock, 1) < k) {
			int tmpIndex = atomicAdd(&assigned, 1);
			if (assignedPoints[tmpIndex] != 0) {
				int NumBlocksChild = dim / threadsXblock;
				if (dim % threadsXblock != 0)
					NumBlocksChild++;
				computeCentroids << <NumBlocksChild, threadsXblock >> > (d_centroids, assignedPoints[tmpIndex], dim, tmpIndex * dim);
				cudaDeviceSynchronize();
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0 && countB >= 2 * NumBlocks && assigned >= k) {
		assigned = 0;
		countB = 0;
		lock = 0;

	}

}

/*
For each point calculate the new centroid and assign it.
	@d_dataPoints: the list of points (device side)
	@d_centroids: the list of centroids (device side)
	@dim: dimension of points
	@length: amount of points
	@k: number of centroids
	@d_convergenceCheck: pointer of the convergence flag. If true the algorithm stops.
	@NumBlocks: the number of blocks which are running this kernel
*/
__global__ void k_means_cuda_device_assign_centroids(double* d_dataPoints, double* d_centroids, int length, int dim, int k, bool* d_convergenceCheck, int NumBlocks) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tid *= (dim + 1);
	if (tid >= length * (dim + 1))
		return;
	int newCentroid = calculateCentroid_device(d_dataPoints, tid, dim, k, d_centroids);

	__syncthreads();

	if ((int)(d_dataPoints[tid + dim]) != newCentroid) {
		atomicAdd(&errorsCounter, 1);
		d_dataPoints[tid + dim] = newCentroid;
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(&countB, 1);
	}

	__syncthreads();

	if (countB >= NumBlocks) {
		if (threadIdx.x < 1024 && atomicAdd(&lock, 1) < 1) {
			countB = 0;
			lock = 0;
			if (errorsCounter <= ((length * convergenceThreshold) / 100)) {
				d_convergenceCheck[0] = true;
			}
			else {
				d_convergenceCheck[0] = false;
			}
			errorsCounter = 0;
		}
	}
}

/*
Main KMEANS call. Host initialization.
	@dataPoints: point to datapoints
	@length: number of points
	@dim: dimension of points (2D, 3D etc.)
	@useParallelism: to use or not OpenMP
	@seed: the seed for the input generator
*/
void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {
	// Randomizer
	std::uniform_real_distribution<> distrib(0, length*100);

	double* h_dataPoints = new double[length * (dim + 1)];
	double* h_centroids = new double[k * dim];
	int* h_assignedPoints = new int[k];

	// 1. Choose the number of clusters(K) and obtain the data points
	if (k <= 0) {
		k = 1;
	}

	//2. Place the centroids c_1, c_2, .....c_k randomly
	float** centroids = new float* [k];
	for (int i = 0; i < k; i++) {
		centroids[i] = new float[dim];
		h_assignedPoints[i] = 0;

		// rand init
		for (int j = 0; j < dim; j++) {
			centroids[i][j] = distrib(seed);	// random clusters
		}
	}

	// Linearization of dataPoints (from nD to 1D) and centroids (from nD to 1D)
	linealizer(h_dataPoints, dataPoints, length, dim + 1);
	linealizer(h_centroids, centroids, k, dim);

	double* d_dataPoints;
	double* d_centroids;

	bool* d_convergenceCheck;
	int* d_assignedPoints;

	int NumBlocks;

	bool convergence = false;
	bool* convergenceCheck = (bool*)malloc(sizeof(bool));
	convergenceCheck[0] = true;

	// device allocation for linearized array
	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(double) * length * (dim + 1)));
	HANDLE_ERROR(cudaMalloc((void**)&d_centroids, sizeof(double) * k * dim));
	HANDLE_ERROR(cudaMalloc((void**)&d_assignedPoints, sizeof(int) * k));
	HANDLE_ERROR(cudaMalloc((void**)&d_convergenceCheck, sizeof(bool)));

	// copy of host linearized array to device
	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints, sizeof(double) * length * (dim + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_centroids, h_centroids, sizeof(double) * k * dim, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_convergenceCheck, convergenceCheck, sizeof(bool), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_assignedPoints, h_assignedPoints, sizeof(int) * k, cudaMemcpyHostToDevice));

	int threadsXblock = 1024;	//number of thread per block
	//loop until convergence is not reached
	while (!convergence) {

		convergenceCheck[0] = true;
		NumBlocks = length / threadsXblock;

		if (length % threadsXblock != 0) {
			NumBlocks += 1;
		}
		//KMEANS call to assign centroids
		k_means_cuda_device_assign_centroids <<< NumBlocks, threadsXblock >>> (d_dataPoints, d_centroids, length, dim, k, d_convergenceCheck, NumBlocks);
		cudaDeviceSynchronize();

		// copy of device convergence array to host
		HANDLE_ERROR(cudaMemcpy(convergenceCheck, d_convergenceCheck, sizeof(bool), cudaMemcpyDeviceToHost));

		convergence = convergenceCheck[0];
		NumBlocks = (length * (dim + 1)) / threadsXblock;

		if ((length * (dim + 1)) % threadsXblock != 0) {
			NumBlocks += 1;
		}

		//memset to zero for centroids and counter of assigned points in device copy
		HANDLE_ERROR(cudaMemset(d_assignedPoints, 0, sizeof(int) * k));
		HANDLE_ERROR(cudaMemset(d_centroids, 0, sizeof(double) * k * dim));

		//KMEANS call to update centroids
		k_means_cuda_device_update_centroids <<< NumBlocks, threadsXblock, k* (dim + 1) * sizeof(double) + k * sizeof(int) >>> (d_dataPoints, d_centroids, d_assignedPoints, length, dim, k, NumBlocks);
		cudaDeviceSynchronize();

	}

	// copy of device linearized array (result) to host
	HANDLE_ERROR(cudaMemcpy(h_dataPoints, d_dataPoints, sizeof(double) * length * (dim + 1), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_centroids, d_centroids, sizeof(double) * k * dim, cudaMemcpyDeviceToHost));

	delinealizer(dataPoints, h_dataPoints, length * (dim + 1), dim);

	// Memory deallocation
	cudaFree(d_dataPoints);
	cudaFree(d_centroids);
	cudaFree(d_assignedPoints);
	cudaFree(d_convergenceCheck);
	free(h_dataPoints);
	free(h_centroids);
	free(h_assignedPoints);
	free(convergenceCheck);
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