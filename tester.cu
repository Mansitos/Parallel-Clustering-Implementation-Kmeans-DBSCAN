/*
Progetto Programmazione su Architetture Parallele - UNIUD 2021
Mansi Andrea & Christian Cagnoni
*/

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include "inputgenerator.h"
#include "utils.h"
#include <chrono>
#include "kmeans.h"
#include "dbscan.h"
#include "kmeansCUDA.h"
#include "dbscanCUDA.h"

using namespace std;
bool checkCorrectness = false;

void checkResCorrectness(float** dataPoints, int numberOfPoints, int dimOfPoints, string algorithm, std::mt19937 seed);

/*
Run a test for the specified algorithm.
	@algorithm: the algorithm to test
	@numberOfPoints: the amout of points 
	@dimOfPoints: specifies the dimension for the points (2D, 3D etc.)
	@repetitions: how much time a tests much be executed (for more accurated results)

	Return: the mean time of execution (on the same input);
*/
chrono::duration<double> runTest(int numberOfPoints, int dimOfPoints, string algorithm, int repetitions, std::mt19937 seed) {
	// random input init
	float** dataPoints = generateRandomInput(numberOfPoints, dimOfPoints);
	chrono::duration<double> time = std::chrono::seconds(0);

	for (int rep = 0; rep < repetitions; rep++) {

		// initialization
		auto start = std::chrono::high_resolution_clock::now();
		auto finish = std::chrono::high_resolution_clock::now();

		// Serial KMEANS
		if (algorithm == "kmeans") {
			k_means(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			finish = std::chrono::high_resolution_clock::now();
		// OpenMP KMEANS
		} else if (algorithm == "kmeans_openmp") {
			k_means(dataPoints, numberOfPoints, dimOfPoints, true, 2, seed);
			finish = std::chrono::high_resolution_clock::now();

			if (checkCorrectness) {
				checkResCorrectness(dataPoints, numberOfPoints, dimOfPoints, algorithm, seed);
			}
		// CUDA KMEANS
		} else if (algorithm == "kmeans_cuda") {
			k_means_cuda_host(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			finish = std::chrono::high_resolution_clock::now();

			if (checkCorrectness) {
				checkResCorrectness(dataPoints, numberOfPoints, dimOfPoints, algorithm, seed);
			}
		// SERIAL DBSCAN
		}
		else if (algorithm == "dbscan") {
			dbscan(dataPoints, numberOfPoints, dimOfPoints, false, seed);
			finish = std::chrono::high_resolution_clock::now();
		// OPENMP DBSCAN
		} else if (algorithm == "dbscan_openmp") {
			dbscan(dataPoints, numberOfPoints, dimOfPoints, true, seed);
			finish = std::chrono::high_resolution_clock::now();

			if (checkCorrectness) {
				checkResCorrectness(dataPoints, numberOfPoints, dimOfPoints, algorithm, seed);
			}
		}
		// CUDA DBSCAN
		else if (algorithm == "dbscan_cuda") {
			dbscan_cuda_host(dataPoints, numberOfPoints, dimOfPoints, false, seed);
			finish = std::chrono::high_resolution_clock::now();

			if (checkCorrectness) {
				checkResCorrectness(dataPoints, numberOfPoints, dimOfPoints, algorithm, seed);
			}
		}

		chrono::duration<double> elapsed = finish - start;
		time += elapsed;

		// clear output (result) column
		clearClusterColumn(dataPoints, numberOfPoints, dimOfPoints);
	}

	delete dataPoints; // input memory clear
	return time/repetitions;
}

/*
	Given an algorithm and its execution result, checks if it is valid. (by executing the serial version)
*/
void checkResCorrectness(float** dataPoints, int numberOfPoints, int dimOfPoints, string algorithm, std::mt19937 seed) {
	// get result column from dataPoints
	int* b = new int[numberOfPoints];	
	#pragma omp parallel for
	for (int i = 0; i < numberOfPoints; i++) {
		b[i] = dataPoints[i][dimOfPoints];
	}

	// clear output (result) column before serial computation
	clearClusterColumn(dataPoints, numberOfPoints, dimOfPoints);

	// execute the serial algorithm with the same input
	if (algorithm == "kmeans_openmp" || algorithm == "kmeans_cuda") {
		k_means(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
	}
	else if (algorithm == "dbscan_openmp" || algorithm == "dbscan_cuda") {
		dbscan(dataPoints, numberOfPoints, dimOfPoints, false, seed);
	}

	int treshold = 1;	// >0 <100 (percentage of allowed errors - different assigned clusters)
	int maxErrs = (numberOfPoints * treshold) / 100;
	int errCounter = 0;
	bool errFlag = false;

	// comparison of the results with error counting
	#pragma omp parallel for
	for (int i = 0; i < numberOfPoints; i++) {
		if (b[i] != (int)(dataPoints[i][dimOfPoints])) {
			#pragma omp atomic
			errCounter++;
			if (errCounter > maxErrs) {	// if too much errors...
				errFlag = true;
			}
		}
	}

	if (errFlag) {
		if (algorithm == "kmeans_openmp" || algorithm == "kmeans_cuda") {
			cout << "ERROR: serial and parallel (" << algorithm << ") version of kmeans mistmatch too much!\n";
		}
		else if (algorithm == "dbscan_openmp" || algorithm == "dbscan_cuda") {
			cout << "ERROR: serial and parallel (" << algorithm << ") version of dbscan mistmatch too much!\n";
		}
	}

	printf("Different clusters between serial and parallel version: %d\n", errCounter);
}

/*
Run an entire tests sessions.
	@saveToCsv: specifies if results have to be saved in a csv file
*/
void runTestSession(bool saveToCsv = false) {
	// how much times a test must be executed (for better accuracy)
	int reps = 10;

	// the lenthts (number of points) that have to be tested
	const int nLenghts = 9;
	int lenghtsToTest[nLenghts] = { 10,50,100,1000,10000,20000,100000,1000000,10000000 };

	// the dimensions (of the points: 2D, 3D etc.) that have to be tested
	const int nDims = 3;
	int dimensionsToTest[nDims] = { 2,3,10 };

	// the algorithms that have to be testeds
	// valid values: kmeans | dbscan | cuda_kmeans | cuda_dbscan | kmeans_openmp | dbscan_openmp
	const int nAlgs = 6;
	string algorithmsToTest[] = { "kmeans","kmeans_openmp","kmeans_cuda","dbscan","dbscan_openmp","dbscan_cuda"};

	// CSV file initialization
	ofstream file("tests.txt");
	if (saveToCsv) {
		string firstline = "index;algorithm;meantime;reps;length;pointsDim\n";
		file << firstline;
	}

	int testIndex = 0;

	std::random_device rd;   // Will be used to obtain a seed for the random number engine
	std::mt19937 seed(rd()); // Standard mersenne_twister_engine seeded with rd()

	// another for with "list of algs to test"
	for (int alg = 0; alg < nAlgs; alg++) {
		cout << "Tested algorithm: " << algorithmsToTest[alg] << "\n";

		for (int dim = 0; dim < nDims; dim++) {
			for (int length = 0; length < nLenghts; length++) {
				chrono::duration<double> meanTime = runTest(lenghtsToTest[length], dimensionsToTest[dim], algorithmsToTest[alg], reps,seed);
				
				testIndex++;

				printf("Reps: %d\n", reps);
				printf("Elements: %d of dim: %d\n", lenghtsToTest[length], dimensionsToTest[dim]);
				std::cout << "Mean time: " << meanTime.count() << "s\n";
				printf("--------------------------------------------\n");

				if (saveToCsv) {
					string newline = "";
					newline.append(std::to_string(testIndex));
					newline.append(";");
					newline.append(algorithmsToTest[alg]);
					newline.append(";");
					newline.append(std::to_string(meanTime.count()));
					newline.append(";");
					newline.append(std::to_string(reps));
					newline.append(";");
					newline.append(std::to_string(lenghtsToTest[length]));
					newline.append(";");
					newline.append(std::to_string(dimensionsToTest[dim]));
					newline.append("\n");

					file << newline;
				}

			}
		}
	}
}

