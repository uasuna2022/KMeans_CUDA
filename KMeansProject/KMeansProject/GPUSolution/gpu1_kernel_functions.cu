#include "gpu1_kernel_functions.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 512

__global__ void calculate_sums_and_counts(const float* __restrict__ d_points, float* __restrict__ d_centroids, 
	int* __restrict__ d_labels, int n, int k, int d, float* __restrict__ d_sums, int* __restrict__ d_counts)
{
	extern __shared__ char shared_memory[];
	int* shared_counts = (int*)shared_memory;
	float* shared_sums = (float*)(shared_counts + k);
	float* shared_centroids = (float*)(shared_sums + k * d);

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < k; i += blockDim.x)
		shared_counts[i] = 0;
	for (int i = tid; i < k * d; i += blockDim.x)
		shared_sums[i] = 0.0F;
	for (int i = tid; i < k * d; i += blockDim.x)
		shared_centroids[i] = d_centroids[i];
	
	__syncthreads();

	if (idx < n)
	{
		float best_distance = FLT_MAX;
		int nearest_cluster = -1;
		for (int i = 0; i < k; i++)
		{
			float dist = 0.0F;
			for (int j = 0; j < d; j++)
			{
				float diff = shared_centroids[k * j + i] - d_points[n * j + idx];
				diff *= diff;
				dist += diff;
			}
			if (dist < best_distance)
			{
				best_distance = dist;
				nearest_cluster = i;
			}
		}

		d_labels[idx] = nearest_cluster;
		atomicAdd(&shared_counts[nearest_cluster], 1);

		for (int j = 0; j < d; j++)
			atomicAdd(&shared_sums[j * k + nearest_cluster], d_points[n * j + idx]);
	}

	__syncthreads();

	for (int i = tid; i < k; i += blockDim.x)
	{
		if (shared_counts[i] > 0)
			atomicAdd(&d_counts[i], shared_counts[i]);
	}
	for (int i = tid; i < k * d; i += blockDim.x)
		atomicAdd(&d_sums[i], shared_sums[i]);
}

__global__ void update_centroids(float* __restrict__ d_centroids, const float* __restrict__ d_sums,
	const int* d_counts, int k, int d, float* __restrict__ d_cluster_deltas)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < k * d)
	{
		int count = d_counts[idx % k];
		if (count > 0)
		{
			float new_value = d_sums[idx] / (float)count;
			float old_value = d_centroids[idx];
			d_centroids[idx] = new_value;
			float diff = new_value - old_value;
			diff *= diff;
			atomicAdd(&d_cluster_deltas[idx % k], diff);
		}
	}
}

void make_iteration(KMeansData* data, int* iteration_number, float* delta)
{
	int N = data->n;
	int K = data->k;
	int D = data->d;

	CHECK_CUDA(cudaMemset(data->d_sums, 0, K * D * sizeof(float)));
	CHECK_CUDA(cudaMemset(data->d_counts, 0, K * sizeof(int)));

	float* d_cluster_deltas;
	CHECK_CUDA(cudaMalloc(&d_cluster_deltas, K * sizeof(float)));
	CHECK_CUDA(cudaMemset(d_cluster_deltas, 0, K * sizeof(float)));

	int blocks_number = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	size_t shared_memory_size = K * sizeof(int) + 2 * K * D * sizeof(float);

	calculate_sums_and_counts << <blocks_number, BLOCK_SIZE, shared_memory_size>> > (
		data->d_points, data->d_centroids, data->d_labels, N, K, D, data->d_sums, data->d_counts);
	cudaDeviceSynchronize();

	int blocks_centroids = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
	update_centroids << <blocks_centroids, BLOCK_SIZE >> > (data->d_centroids, data->d_sums, data->d_counts, K, D, d_cluster_deltas);

	std::vector<float> h_cluster_deltas;
	h_cluster_deltas.resize(K);
	CHECK_CUDA(cudaMemcpy(h_cluster_deltas.data(), d_cluster_deltas, K * sizeof(float), cudaMemcpyDeviceToHost));

	float total_shift = 0.0F;
	for (int i = 0; i < K; i++)
		total_shift += std::sqrt(h_cluster_deltas[i]);

	*delta = total_shift;
	(*iteration_number)++;

	cudaFree(d_cluster_deltas);
	cudaDeviceSynchronize();
}