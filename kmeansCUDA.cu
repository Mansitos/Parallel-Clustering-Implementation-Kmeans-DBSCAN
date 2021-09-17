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

__device__ bool convergenceCheck = true;
__device__ unsigned int countB = 0;

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

static __inline__ __device__ bool atomicCASBool(bool* address, bool compare, bool val)
{
	unsigned long long addr = (unsigned long long)address;
	unsigned pos = addr & 3;  // byte position within the int
	int* int_addr = (int*)(addr - pos);  // int-aligned address
	int old = *int_addr, assumed, ival;

	bool current_value;

	do
	{
		current_value = (bool)(old & ((0xFFU) << (8 * pos)));

		if (current_value != compare) // If we expected that bool to be different, then
			break; // stop trying to update it and just return it's current value

		assumed = old;
		if (val)
			ival = old | (1 << (8 * pos));
		else
			ival = old & (~((0xFFU) << (8 * pos)));
		old = atomicCAS(int_addr, assumed, ival);
	} while (assumed != old);

	return current_value;
}

__device__ bool AtomicBool(bool* address, bool val)
{
	// Create an initial guess for the value stored at *address.
	bool guess = *address;
	bool oldValue = atomicCASBool(address, guess, val);

	// Loop while the guess is incorrect.
	while (oldValue != guess)
	{
		guess = oldValue;
		oldValue = atomicCASBool(address, guess, val);
	}
	return oldValue;
}

__device__ double euclideanDistance_device(double* dataPoint, int tid, double* centroid, int index, int dim) {
	double sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += pow((dataPoint[tid + i] - centroid[index + i]), 2);
	}
	double distance = sqrt(sum);
	return distance;
}

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
		else if (distance < minDistance ){
			nearestCentroidIndex = i;
			minDistance = distance;
		}
	}
	return nearestCentroidIndex;
}

__global__ void k_means_cuda_device_update_centroids(double* d_dataPoints, double* d_centroids, int* assignedPoints, int length, int dim, int k, int NumBlocks) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length * (dim + 1))
		return;
	if (threadIdx.x < k && countB == 0)
		assignedPoints[tid] = 0;
	int tmp = tid * k;
	int tmpId = threadIdx.x * k;
	if (tmpId < k * dim && countB == 0) {
		for (int i = tmpId; i < tmpId + dim; i++)
			d_centroids[i] = 0;
	}

	if (threadIdx.x == 0)
		atomicAdd(&countB, 1);

	__syncthreads();

	extern __shared__ double s[];
	double* b_centroids = s;
	int* b_assignedPoints = (int*)&b_centroids[k * (dim + 1)];
	if (threadIdx.x == 0)
		for (int i = 0; i < k; i++) {
			b_assignedPoints[i] = 0;
			for (int j = 0; j < dim; j++)
				b_centroids[(i * dim) + j] = 0;
		}
	__syncthreads();

	if (tid % (dim + 1) == 0) {//prima cordinata
		atomicAddDouble(&b_centroids[((int)d_dataPoints[tid + dim] * k)], d_dataPoints[tid]);
	}
	else if ((tid + 1) % (dim + 1) != 0) {//seconda
		atomicAddDouble(&b_centroids[((int)d_dataPoints[tid + (tid % (dim + 1))] * k) + (tid % (dim + 1))], d_dataPoints[tid]);
	}
	else {//cluster
		atomicAdd(&b_assignedPoints[(int)d_dataPoints[tid]], 1);
	}

	__syncthreads();

	if (threadIdx.x == 0)
		for (int i = 0; i < k; i++) {
			atomicAdd(&assignedPoints[i], b_assignedPoints[i]);
			for (int j = 0; j < dim; j++)
				atomicAddDouble(&d_centroids[(i * dim) + j], b_centroids[(i * dim) + j]);
		}

	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(&countB, 1);
	}

	__syncthreads();

	if (tmp < k * dim) {
		while (countB != 2 * NumBlocks) {
		}
		if (assignedPoints[tmp / k] != 0)
			for (int i = tmp; i < tmp + dim; i++)
				d_centroids[i] /= assignedPoints[tmp / k];
		countB = 0;
	}
}

__global__ void k_means_cuda_device_assign_centroids(double* d_dataPoints, double* d_centroids, int length, int dim, int k, bool* d_convergenceCheck, int NumBlocks) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tid *= (dim + 1);
	if (tid >= length * (dim + 1))
		return;
	int newCentroid = calculateCentroid_device(d_dataPoints, tid, dim, k, d_centroids);
	__syncthreads();
	if ((int)(d_dataPoints[tid + dim]) != newCentroid) {
		AtomicBool(&convergenceCheck, false);
		d_dataPoints[tid + dim] = newCentroid;
	}
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(&countB, 1);

	__syncthreads();
	if (tid == 0) {
		while (countB != NumBlocks) {
		}
		countB = 0;
		d_convergenceCheck[0] = convergenceCheck;
		convergenceCheck = true;
	}
}

void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {
	// Randomizer
	std::uniform_real_distribution<> distrib(0, 10);

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

	linealizer(h_dataPoints, dataPoints, length, dim + 1);
	linealizer(h_centroids, centroids, k, dim);

	double* d_dataPoints;

	double* d_centroids;

	bool* d_convergenceCheck;

	int* d_assignedPoints;

	int NumBlocks;

	bool convergence = false;

	bool* convergenceCheck = new bool[1];

	convergenceCheck[0] = true;

	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(double) * length * (dim + 1)));
	HANDLE_ERROR(cudaMalloc((void**)&d_centroids, sizeof(double) * k * dim));
	HANDLE_ERROR(cudaMalloc((void**)&d_assignedPoints, sizeof(int) * k));
	HANDLE_ERROR(cudaMalloc((void**)&d_convergenceCheck, sizeof(bool)));

	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints, sizeof(double) * length * (dim + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_centroids, h_centroids, sizeof(double) * k * dim, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_convergenceCheck, convergenceCheck, sizeof(bool), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_assignedPoints, h_assignedPoints, sizeof(int) * k, cudaMemcpyHostToDevice));

	while (!convergence) {

		convergenceCheck[0] = true;

		NumBlocks = length / 256;

		if (length % 256 != 0) {
			NumBlocks += 1;
		}

		k_means_cuda_device_assign_centroids << < NumBlocks, 256 >> > (d_dataPoints, d_centroids, length, dim, k, d_convergenceCheck, NumBlocks);
		cudaDeviceSynchronize();

		HANDLE_ERROR(cudaMemcpy(convergenceCheck, d_convergenceCheck, sizeof(bool), cudaMemcpyDeviceToHost));

		convergence = convergenceCheck[0];

		NumBlocks = (length * (dim + 1)) / 256;

		if ((length * (dim + 1)) % 256 != 0) {
			NumBlocks += 1;
		}

		k_means_cuda_device_update_centroids << < NumBlocks, 256, k* (dim + 1) * sizeof(double) + k * sizeof(int) >> > (d_dataPoints, d_centroids, d_assignedPoints, length, dim, k, NumBlocks);
		cudaDeviceSynchronize();

	}

	HANDLE_ERROR(cudaMemcpy(h_dataPoints, d_dataPoints, sizeof(double) * length * (dim + 1), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_centroids, d_centroids, sizeof(double) * k * dim, cudaMemcpyDeviceToHost));

	/*printf("DATAPOINTS\n");
	int count = 1;
	for (int i = 0; i < length * (dim + 1); i++) {
		printf("%f ", h_dataPoints[i]);
		if (count == (dim+1)) {
			printf("\n");
			count = 0;
		}
		count++;
	}
	printf("CENTROIDS\n");
	for (int i = 0; i < k * dim; i++) {
		printf("%f ", h_centroids[i]);
		if (count == dim) {
			printf("\n");
			count = 0;
		}
		count++;
	}*/

	delinealizer(dataPoints, h_dataPoints, length * (dim + 1), dim);

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