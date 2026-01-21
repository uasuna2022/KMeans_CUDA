#include "gpu2_kernel_functions.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cfloat>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 512

__global__ void find_nearest_cluster(const float* __restrict__ d_points, const float* __restrict__ d_centroids, 
	int* __restrict__ d_labels, int n, int k, int d, int* __restrict__ d_changes_count)
{
	__shared__ int shared_changes;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x == 0)
		shared_changes = 0;
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
				float diff = d_centroids[k * j + i] - d_points[n * j + idx];
				diff *= diff;
				dist += diff;
			}
			if (dist < best_distance)
			{
				best_distance = dist;
				nearest_cluster = i;
			}
		}

		if (nearest_cluster != d_labels[idx])
		{
			d_labels[idx] = nearest_cluster;
			atomicAdd(&shared_changes, 1);
		}
	}

	__syncthreads();

	if (threadIdx.x == 0 && shared_changes > 0)
	{
		atomicAdd(d_changes_count, shared_changes);
	}
}

__global__ void calculate_centroids_and_delta(float* __restrict__ d_centroids, const float* __restrict__ d_sums,
	const int* __restrict__ d_counts, int k, int d, float* __restrict__ d_deltas)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < k * d)
	{
		int cluster_id = idx % k;
		int count = d_counts[cluster_id];

		if (count > 0)
		{
			float sum = d_sums[idx];
			float old_val = d_centroids[idx];
			float new_val = sum / (float)count;

			d_centroids[idx] = new_val;

			float diff = new_val - old_val;
			diff *= diff;
			atomicAdd(&d_deltas[cluster_id], diff);
		}
	}
}

void make_iteration_2(KMeansData* data, int* iteration_number, float* delta, int* points_changed)
{
	int N = data->n;
	int K = data->k;
	int D = data->d;

	CHECK_CUDA(cudaMemset(data->d_sums, 0, K * D * sizeof(float)));
	CHECK_CUDA(cudaMemset(data->d_counts, 0, K * sizeof(int)));
	CHECK_CUDA(cudaMemset(data->d_changes_count, 0, sizeof(int)));
	CHECK_CUDA(cudaMemset(data->d_deltas, 0, K * sizeof(float)));

	int blocks_number = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	find_nearest_cluster << <blocks_number, BLOCK_SIZE >> > (data->d_points, data->d_centroids, data->d_labels, N, K, D, data->d_changes_count);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	static thrust::device_vector<int> th_labels_copy;
	static thrust::device_vector<int> th_indices;

	static thrust::device_vector<float> th_coords_gathered;
	static thrust::device_vector<int> th_unique_keys;
	static thrust::device_vector<float> th_reduced_sums;    
	static thrust::device_vector<int> th_reduced_counts;

	if (th_labels_copy.size() != N) 
	{
		th_labels_copy.resize(N);
		th_indices.resize(N);
		th_coords_gathered.resize(N);
		th_unique_keys.resize(K);
		th_reduced_sums.resize(K);
		th_reduced_counts.resize(K);
	}

	thrust::device_ptr<int> dev_labels(data->d_labels);
	thrust::copy(dev_labels, dev_labels + N, th_labels_copy.begin());

	thrust::sequence(th_indices.begin(), th_indices.end());

	thrust::sort_by_key(th_labels_copy.begin(), th_labels_copy.end(), th_indices.begin());

	auto end_counts = thrust::reduce_by_key(th_labels_copy.begin(), th_labels_copy.end(), thrust::make_constant_iterator(1),
		th_unique_keys.begin(), th_reduced_counts.begin());

	thrust::device_ptr<int> dev_counts(data->d_counts);
	thrust::scatter(th_reduced_counts.begin(), end_counts.second, th_unique_keys.begin(), dev_counts);

	thrust::device_ptr<float> dev_points(data->d_points);
	thrust::device_ptr<float> dev_sums(data->d_sums);

	for (int j = 0; j < D; j++)
	{
		thrust::gather(th_indices.begin(), th_indices.end(), dev_points + (j * N), th_coords_gathered.begin());

		auto end_sums = thrust::reduce_by_key(th_labels_copy.begin(), th_labels_copy.end(), th_coords_gathered.begin(),                  
			th_unique_keys.begin(), th_reduced_sums.begin());

		thrust::scatter(th_reduced_sums.begin(), end_sums.second, th_unique_keys.begin(), dev_sums + (j * K));
	}
	CHECK_CUDA(cudaDeviceSynchronize());

	int blocks_update = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
	calculate_centroids_and_delta << <blocks_update, BLOCK_SIZE >> > (data->d_centroids, data->d_sums, data->d_counts,
		K, D, data->d_deltas);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	std::vector<float> h_deltas(K);
	CHECK_CUDA(cudaMemcpy(h_deltas.data(), data->d_deltas, K * sizeof(float), cudaMemcpyDeviceToHost));

	float total_shift = 0.0f;
	for (int i = 0; i < K; i++) 
		total_shift += sqrt(h_deltas[i]);
	*delta = total_shift;

	CHECK_CUDA(cudaMemcpy(points_changed, data->d_changes_count, sizeof(int), cudaMemcpyDeviceToHost));

	(*iteration_number)++;
}