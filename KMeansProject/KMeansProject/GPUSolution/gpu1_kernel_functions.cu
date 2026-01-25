#include "gpu1_kernel_functions.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 512

// This kernel function performs the assignment step of K-Means algo.
// Firstly, it finds the nearest cluster for each point (running N CUDA threads).
// Then it accumulates the sum of coordinates and the count of points for each cluster.
// Shared memory is used here to perform partial reductions within a block of threads.
// The main goal of shared memory usage is to reduce global memory atomic operations 
// and speed up the calculations (as shared memory is faster).
__global__ void calculate_sums_and_counts(const float* __restrict__ d_points, float* __restrict__ d_centroids, 
	int* __restrict__ d_labels, int n, int k, int d, double* __restrict__ d_sums, int* __restrict__ d_counts,
	int* __restrict__ d_changes_count)
{
	// Initialize shared memory (it will store accumulated sums for every cluster,
	// the counter for points that changed their cluster during current iteration
	// and number of points in each cluster).
	extern __shared__ char shared_memory[];
	double* shared_sums = (double*)shared_memory;
	int* shared_counts = (int*)&shared_sums[k * d];

	__shared__ int shared_block_changes_counter;

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0) 
		shared_block_changes_counter = 0;
	for (int i = tid; i < k; i += blockDim.x)
		shared_counts[i] = 0;
	for (int i = tid; i < k * d; i += blockDim.x)
		shared_sums[i] = 0.0;
	
	__syncthreads();

	if (idx < n)
	{
		// Every CUDA thread finds the nearest cluster
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

		// If a point changed its cluster, we update the info about it 
		// and increase appropriate counters
		if (nearest_cluster != d_labels[idx])
		{
			d_labels[idx] = nearest_cluster;
			atomicAdd(&shared_block_changes_counter, 1);
		}

		atomicAdd(&shared_counts[nearest_cluster], 1);

		// Here we add the coords of current point to the appropriate cluster 
		// counter, using atomic operations to avoid posiible race condition.
		for (int j = 0; j < d; j++)
		{
			double value = (double)d_points[n * j + idx];
			atomicAdd(&shared_sums[j * k + nearest_cluster], value);
		}
	}

	__syncthreads();

	// After all "point" threads finished their job, we can accumulate 
	// data from all shared memory blocks to the global memory.
	if (tid == 0 && shared_block_changes_counter > 0)
		atomicAdd(d_changes_count, shared_block_changes_counter);
	for (int i = tid; i < k; i += blockDim.x)
	{
		if (shared_counts[i] > 0)
			atomicAdd(&d_counts[i], shared_counts[i]);
	}
	for (int i = tid; i < k * d; i += blockDim.x)
		atomicAdd(&d_sums[i], shared_sums[i]);
}

// This kernel function performs the Update Step of K-Means algo.
// Divides the accumulated sums by the count of points to get new centroid coords.
__global__ void update_centroids(float* __restrict__ d_centroids, const double* __restrict__ d_sums,
	const int* d_counts, int k, int d)
{
	// Here we run one thread per centroid coordinate (k * d in summary).
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < k * d)
	{
		int count = d_counts[idx % k];
		if (count > 0)
		{
			float new_value = (float)(d_sums[idx] / (double)count);
			d_centroids[idx] = new_value;
		}
	}
}

// This function simulates one iteration of K-Means algo. 
void make_iteration(KMeansData* data, int* iteration_number, int* points_changed)
{
	int N = data->n;
	int K = data->k;
	int D = data->d;

	// Reset intermediate buffers before the iteration.
	CHECK_CUDA(cudaMemset(data->d_sums, 0, K * D * sizeof(double)));
	CHECK_CUDA(cudaMemset(data->d_counts, 0, K * sizeof(int)));
	CHECK_CUDA(cudaMemset(data->d_changes_count, 0, sizeof(int)));

	// Calculate the grid dimensions and the size of required shared memory per block.
	int blocks_number = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	size_t shared_memory_size = K * sizeof(int) + K * D * sizeof(double);

	// Launch Assignment Kernel
	calculate_sums_and_counts << <blocks_number, BLOCK_SIZE, shared_memory_size>> > (
		data->d_points, data->d_centroids, data->d_labels, N, K, D, data->d_sums, data->d_counts, data->d_changes_count);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	// Then launch Update Kernel
	int blocks_centroids = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
	update_centroids << <blocks_centroids, BLOCK_SIZE >> > (data->d_centroids, data->d_sums, data->d_counts, K, D);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	// Copy the change counter back to CPU. (There is no need to copy all other data
	// as the algo isn't finished yet. That process will happen once after the algo
	// stops).
	int h_changes = 0;
	CHECK_CUDA(cudaMemcpy(&h_changes, data->d_changes_count, sizeof(int), cudaMemcpyDeviceToHost));
	*points_changed = h_changes;

	(*iteration_number)++;
	cudaDeviceSynchronize();
}