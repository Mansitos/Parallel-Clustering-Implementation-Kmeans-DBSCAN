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

__device__ bool convergenceCheck=true;
__device__ unsigned int countB = 0;

__device__ float euclideanDistance_device(float* dataPoint, int tid, float* centroid,int index, int dim) {
	float sum = 0;
	for (int i = 0; i < dim; i++) {
		sum += pow((dataPoint[tid+i] - centroid[index+i]), 2);
	}
	float distance = sqrt(sum);
	return distance;
}

__device__ int calculateCentroid_device(float* dataPoint,int tid, int dim, int k, float* centroids) {
	int nearestCentroidIndex = dataPoint[tid+dim];
	bool firstIteration = true;
	float minDistance = 0;
	for (int i = 0; i < k; i++) {
		float distance = euclideanDistance_device(dataPoint, tid, centroids,i*dim, dim);
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

__global__ void k_means_cuda_device_update_centroids(float* d_dataPoints, float* d_centroids,int* assignedPoints, int length, int dim, int k,int NumBlocks) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x == 0)
		countB = 0;
	if (tid >= length * (dim + 1))
		return;
	if (threadIdx.x < k)
		assignedPoints[threadIdx.x] = 0;
	int tmp = tid * k;
	int idTmp = threadIdx.x * k;
	if (idTmp < k * dim) {
		for (int i = idTmp; i < idTmp + dim; i++)
			d_centroids[i] = 0;
	}
	__syncthreads();
	extern __shared__ float s[];
	float* b_centroids=s;
	int* b_assignedPoints=(int*)&b_centroids[k*(dim+1)];
	if(threadIdx.x==0)
		for (int i = 0; i < k; i++) {
			b_assignedPoints[i] = 0;
			for (int j = 0; j < dim; j++)
				b_centroids[(i*dim) + j] = 0;
		}
	__syncthreads();

	if (tid % (dim + 1) == 0) {//prima cordinata
		atomicAdd(&b_centroids[((int)d_dataPoints[tid + dim] * k)],d_dataPoints[tid]);
	} else if ((tid + 1) % (dim + 1) != 0) {//seconda
		atomicAdd(&b_centroids[((int)d_dataPoints[tid + (tid % (dim + 1))] * k) + (tid % (dim + 1))],d_dataPoints[tid]);
	} else {//cluster
		atomicAdd(&b_assignedPoints[(int)d_dataPoints[tid]],1);
	}

	__syncthreads();

	if(threadIdx.x==0)
		for (int i = 0; i < k; i++) {
			atomicAdd(&assignedPoints[i], b_assignedPoints[i]);
			for (int j = 0; j < dim; j++)
				atomicAdd(&d_centroids[(i*dim) + j], b_centroids[(i*dim) + j]);
		}

	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(&countB, 1);

	__syncthreads();


	if (tmp < k * dim) {
		while (countB != NumBlocks) {
		}
		if (assignedPoints[tmp / k] != 0)
			for (int i = tmp; i < tmp + dim; i++)
				d_centroids[i] /= assignedPoints[tmp / k];
	}
}

__global__ void k_means_cuda_device_assign_centroids(float* d_dataPoints, float* d_centroids, int length, int dim, int k,bool* d_convergenceCheck) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tid *= (dim + 1);
	if (tid >= length * (dim + 1))
		return;
	convergenceCheck = true;
	int newCentroid = calculateCentroid_device(d_dataPoints,tid, dim, k, d_centroids);
	__syncthreads();
	if (d_dataPoints[tid + dim] != newCentroid) {
		convergenceCheck = false;
		d_dataPoints[tid + dim] = newCentroid;
	}
	__syncthreads();
	if (tid == 0)
		d_convergenceCheck[0] = convergenceCheck;
}

void k_means_cuda_host(float** dataPoints, int length, int dim, bool useParallelism, int k, std::mt19937 seed) {

	// Randomizer
	std::uniform_real_distribution<> distrib(0, 10);

	float* h_dataPoints =new float[length*(dim+1)];
	float* h_centroids=new float[k*dim];
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

	linealizer(h_dataPoints, dataPoints,length,dim+1);
	linealizer(h_centroids, centroids, k, dim);

	float* d_dataPoints;

	float* d_centroids;

	bool* d_convergenceCheck;

	int* d_assignedPoints;

	int NumBlocks;

	bool convergence = false;

	bool *convergenceCheck=new bool[1];

	convergenceCheck[0] = true;

	HANDLE_ERROR(cudaMalloc((void**)&d_dataPoints, sizeof(float) * length * (dim+1)));
	HANDLE_ERROR(cudaMalloc((void**)&d_centroids, sizeof(float) * k * dim));
	HANDLE_ERROR(cudaMalloc((void**)&d_assignedPoints, sizeof(int) * k ));
	HANDLE_ERROR(cudaMalloc((void**)&d_convergenceCheck, sizeof(bool)));

	HANDLE_ERROR(cudaMemcpy(d_dataPoints, h_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_centroids, h_centroids, sizeof(float) * k * dim, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_convergenceCheck, convergenceCheck, sizeof(bool), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_assignedPoints, h_assignedPoints, sizeof(int)*k, cudaMemcpyHostToDevice));

	while (!convergence) {

		convergenceCheck[0] = true;

		NumBlocks = length / 256;

		if (length % 256 != 0) {
			NumBlocks += 1;
		}

		k_means_cuda_device_assign_centroids << < NumBlocks, 256 >> > (d_dataPoints, d_centroids, length, dim, k,d_convergenceCheck);
		cudaDeviceSynchronize();

		HANDLE_ERROR(cudaMemcpy(convergenceCheck, d_convergenceCheck, sizeof(bool), cudaMemcpyDeviceToHost));

		convergence = convergenceCheck[0];

		NumBlocks = (length*(dim+1)) / 256;

		if ((length * (dim + 1)) % 256 != 0) {
			NumBlocks += 1;
		}

		k_means_cuda_device_update_centroids << < NumBlocks, 256,k*(dim+1)*sizeof(float)+k*sizeof(int)>> > (d_dataPoints, d_centroids, d_assignedPoints, length, dim, k, NumBlocks);
		cudaDeviceSynchronize();
	}

	HANDLE_ERROR(cudaMemcpy(h_dataPoints, d_dataPoints, sizeof(float) * length * (dim + 1), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_centroids, d_centroids, sizeof(float) * k * dim, cudaMemcpyDeviceToHost));

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

	delinealizer(dataPoints,h_dataPoints,length*(dim+1),dim);

	cudaFree(d_dataPoints);
	cudaFree(d_centroids);
	free(h_dataPoints);
	free(h_centroids);

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