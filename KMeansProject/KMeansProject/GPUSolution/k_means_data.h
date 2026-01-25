#pragma once
#include <vector>
#include "cuda_runtime.h"
#include <iostream>

#define CHECK_CUDA(error) do { cudaError_t err = error; \
	 if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
		exit(1); \
	 } } while (0) 

// KMeansData class is created to hold all the necessary data for algo inside one object. 
// Contains some helper methods like allocate_memory and free_memory (no explanation needed here)
// and fill_gpu_data method, which copies some data from host to device and initializes the rest of
// objects with zeros.
class KMeansData {
public:
	int n;
	int d;
	int k;

	float* d_points;
	float* d_centroids;
	int* d_labels;
	double* d_sums;
	int* d_counts;
	int* d_changes_count;

	KMeansData(int n, int k, int d);
	~KMeansData();

	void fill_gpu_data(const std::vector<float>& h_points, const std::vector<float>& h_centroids);
private:
	void allocate_memory();
	void free_memory();
};

