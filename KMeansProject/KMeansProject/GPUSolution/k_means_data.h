#pragma once
#include <vector>
#include "cuda_runtime.h"
#include <iostream>

#define CHECK_CUDA(error) do { cudaError_t err = error; \
	 if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
		exit(1); \
	 } } while (0) 

class KMeansData {
public:
	int n;
	int d;
	int k;

	float* d_points;
	float* d_centroids;
	int* d_labels;
	float* d_sums;
	int* d_counts;
	float* d_old_centroids;

	KMeansData(int n, int k, int d);
	~KMeansData();

	void fill_gpu_data(const std::vector<float>& h_points, const std::vector<float>& h_centroids);
private:
	void allocate_memory();
	void free_memory();
};

