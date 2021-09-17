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
#include "kmeans.h"
#include "inputgenerator.h"
#include "utils.h"
#include <chrono>
#include "dbscan.h"
#include "kmeansCUDA.h"
#include "dbscanCUDA.h"

using namespace std;

/*
Run a test for the specified algorithm.
	@algorithm: the algorithm to test
	@numberOfPoints: the amout of points 
	@dimOfPoints: specifies the dimension for the points (2D, 3D etc.)
	@repetitions: how much time a tests much be executed (for more accurated results)

	Return: the mean time of execution (on the same input);
*/
chrono::duration<double> runTest(int numberOfPoints, int dimOfPoints, string algorithm, int repetitions, std::mt19937 seed) {
	float** dataPoints = generateRandomInput(numberOfPoints, dimOfPoints);
	chrono::duration<double> time = std::chrono::seconds(0);

	for (int rep = 0; rep < repetitions; rep++) {
		auto start = std::chrono::high_resolution_clock::now();
		auto finish = std::chrono::high_resolution_clock::now();
		if (algorithm == "kmeans") {
			k_means(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			finish = std::chrono::high_resolution_clock::now();
		} else if (algorithm == "kmeans_openmp") {
			k_means(dataPoints, numberOfPoints, dimOfPoints, true, 2, seed);
			finish = std::chrono::high_resolution_clock::now();
			int* b = new int[numberOfPoints];
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				b[i] = dataPoints[i][dimOfPoints];
			}
			k_means(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			int treshold = 1;
			int err = (numberOfPoints * treshold) / 100;
			int check = 0;
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				if (b[i] != (int)(dataPoints[i][dimOfPoints])) {
					if (++check > err) {
						printf("ERROR: serial and parallel (openmp) version of kmeans have several results\n");
						break;
					}
				}
			}
			if (check <= err)
				printf("Total errors (less than 1%c) :%d\n", (char)37,check);
		} else if (algorithm == "kmeans_cuda") {
			k_means_cuda_host(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			finish = std::chrono::high_resolution_clock::now();
			int* b=new int[numberOfPoints];
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				b[i] = dataPoints[i][dimOfPoints];
			}
			k_means(dataPoints, numberOfPoints, dimOfPoints, false, 2, seed);
			int treshold = 1;
			int err = (numberOfPoints * treshold) / 100;
			int check = 0;
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				if (b[i] != (int)(dataPoints[i][dimOfPoints])) {
					if (++check > err) {
						printf("ERROR: serial and parallel (cuda) version of kmeans have several results\n");
						break;
					}
				}
			}
			if (check <= err)
				printf("Total errors (less than 1%c) :%d\n", (char)37, check);
		} else if (algorithm == "kmeans_cuda_openmp") {

		} else if (algorithm == "dbscan") {
			dbscan(dataPoints, numberOfPoints, dimOfPoints, false, seed);
			finish = std::chrono::high_resolution_clock::now();
		} else if (algorithm == "dbscan_openmp") {
			dbscan(dataPoints, numberOfPoints, dimOfPoints, true, seed);
			finish = std::chrono::high_resolution_clock::now();
			int* b = new int[numberOfPoints];
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				b[i] = dataPoints[i][dimOfPoints];
			}
			dbscan(dataPoints, numberOfPoints, dimOfPoints, false,seed);
			int treshold = 1;
			int err = (numberOfPoints * treshold) / 100;
			int check = 0;
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				if (b[i] != (int)(dataPoints[i][dimOfPoints])) {
					if (++check > err) {
						printf("ERROR: serial and parallel (openmp) version of dbscan have several results\n");
						break;
					}
				}
			}
			if (check <= err)
				printf("Total errors (less than 1%c) :%d\n", (char)37, check);
		} else if (algorithm == "dbscan_cuda") {
			dbscan_cuda_host(dataPoints, numberOfPoints, dimOfPoints, false, seed);
			finish = std::chrono::high_resolution_clock::now();
			int* b = new int[numberOfPoints];
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				b[i] = dataPoints[i][dimOfPoints];
			}
			dbscan(dataPoints, numberOfPoints, dimOfPoints, false, seed);
			int treshold = 1;
			int err = (numberOfPoints * treshold) / 100;
			int check = 0;
			#pragma omp parallel for
			for (int i = 0; i < numberOfPoints; i++) {
				if (b[i] != (int)(dataPoints[i][dimOfPoints])) {
					if (++check > err) {
						printf("ERROR: serial and parallel (cuda) version of dbscan have several results\n");
						break;
					}
				}
			}
			if (check <= err)
				printf("Total errors (less than 1%c) :%d\n", (char)37, check);
		} else if (algorithm == "dbscan_cuda_openmp") {

		}
		chrono::duration<double> elapsed = finish - start;
		time += elapsed;
		clearClusterColumn(dataPoints, numberOfPoints, dimOfPoints);
	}
	delete dataPoints; // memory clear
	return time/repetitions;
}

/*
Run an entire tests sessions.
	@saveToCsv: specifies if results have to be saved in a csv file
*/
void runTestSession(bool saveToCsv = false) {
	// how much times a test must be executed (for better accuracy)
	int reps = 1;//10;

	// the lenthts (number of points) that have to be tested
	const int nLenghts = 9;
	int lenghtsToTest[nLenghts] = {10, 50, 100, 1000, 10000,100000,500000,1000000,10000000};

	// the dimensions (of the points: 2D, 3D etc.) that have to be tested
	const int nDims = 1;
	int dimensionsToTest[nDims] = { 2 };

	// the algorithms that have to be testeds
	// valid values: kmeans | dbscan | cuda_kmeans | cuda_dbscan | kmeans_openmp | dbscan_openmp
	const int nAlgs = 3;//6;
	string algorithmsToTest[] = { "dbscan","dbscan_openmp","dbscan_cuda" };//"kmeans","kmeans_openmp","kmeans_cuda","dbscan","dbscan_openmp","dbscan_cuda"};

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
			for (int length = 0; length < 5; length++) {
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

