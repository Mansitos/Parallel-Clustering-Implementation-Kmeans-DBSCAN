/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
#include "inputgenerator.h"
#include <random>
#include <iostream>
#include "utils.h"
#include "tester.h"

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char* file, int line);

int main() {
	runTestSession(true);
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

