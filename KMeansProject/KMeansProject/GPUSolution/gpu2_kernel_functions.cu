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

// Unlike GPU1, this kernel here only assigns labels and counts changes.
// It calculates distances like the one inside GPU1, but doesn't calculate sums.
__global__ void find_nearest_cluster(const float* __restrict__ d_points, const float* __restrict__ d_centroids, 
	int* __restrict__ d_labels, int n, int k, int d, int* __restrict__ d_changes_count)
{
	// So only one counter is required to be allocated in shared memory -
	// the counter of points that changed their cluster. 
	__shared__ int shared_changes;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x == 0)
		shared_changes = 0;
	__syncthreads();

	if (idx < n)
	{
		// Every CUDA thread find the nearest cluster for a certain point
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
		// and increase a counter.
		if (nearest_cluster != d_labels[idx])
		{
			d_labels[idx] = nearest_cluster;
			atomicAdd(&shared_changes, 1);
		}
	}

	__syncthreads();

	// After all threads finished their job, we just accumulate the overall counter
	// from many shared memory blocks to the global memory.
	if (threadIdx.x == 0 && shared_changes > 0)
	{
		atomicAdd(d_changes_count, shared_changes);
	}
}

// The same kernel function as in GPU1 - just finalizes the centroid update by dividing
// summed coords by the count.
__global__ void calculate_centroids_and_delta(float* __restrict__ d_centroids, const double* __restrict__ d_sums,
	const int* __restrict__ d_counts, int k, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < k * d)
	{
		int cluster_id = idx % k;
		int count = d_counts[cluster_id];

		if (count > 0)
		{
			double sum = d_sums[idx];
			float new_val = (float)(sum / (double)count);
			d_centroids[idx] = new_val;
		}
	}
}

// This function simulates one iteration of K-Means algo, but unlike in GPU1, 
// where it's literally just call of 2 kernel functions, here some complicated
// thrust operations are used. Details inside the function -->
void make_iteration_2(KMeansData* data, int* iteration_number, int* points_changed)
{
	int N = data->n;
	int K = data->k;
	int D = data->d;

	// Reset intermediate buffers before the iteration.
	CHECK_CUDA(cudaMemset(data->d_sums, 0, K * D * sizeof(double)));
	CHECK_CUDA(cudaMemset(data->d_counts, 0, K * sizeof(int)));
	CHECK_CUDA(cudaMemset(data->d_changes_count, 0, sizeof(int)));

	// Calculate the grid dimensions.
	int blocks_number = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Assign step - runs the custom kernel to find the nearest cluster for each point.
	find_nearest_cluster << <blocks_number, BLOCK_SIZE >> > (data->d_points, data->d_centroids, 
		data->d_labels, N, K, D, data->d_changes_count);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	// Then thrust section starts. First of all we use static vectors to avoid
	// reallocation every iteration.
	static thrust::device_vector<int> th_labels_copy;
	static thrust::device_vector<int> th_indices;

	static thrust::device_vector<double> th_coords_gathered;
	static thrust::device_vector<int> th_unique_keys;
	static thrust::device_vector<double> th_reduced_sums;    
	static thrust::device_vector<int> th_reduced_counts;

	// Resize vectors on the first iteration.
	if (th_labels_copy.size() != N) 
	{
		th_labels_copy.resize(N);
		th_indices.resize(N);
		th_coords_gathered.resize(N);
		th_unique_keys.resize(K);
		th_reduced_sums.resize(K);
		th_reduced_counts.resize(K);
	}

	// Here we wrap raw pointers with thrust device pointers.
	thrust::device_ptr<int> dev_labels(data->d_labels);

	// Copy labels to a temporary vector for sorting, as we don't wanna 
	// sort original labels array. Now calculated labels are situated 
	// inside th_labels_copy (i.e.: 2 4 1 2 3 1 4 2 0 3 0)
	thrust::copy(dev_labels, dev_labels + N, th_labels_copy.begin());

	// Fill indices with 0, 1, 2, ..., N-1
	thrust::sequence(th_indices.begin(), th_indices.end());

	// Sorts labels and indices together.
	thrust::sort_by_key(th_labels_copy.begin(), th_labels_copy.end(), th_indices.begin());
	
	// After this the arrays look like this:
	// th_labels_copy: 0 0 1 1 2 2 2 3 3 4 4 7
	// th_indices:    8 10 2 5 0 3 7 4 9 1 6 11

	// Then we run reduce_by_key function. It sums up the '1's for each identical key 
	// (cluster label). In fact we just count the number of points associated with
	// each cluster.
	auto end_counts = thrust::reduce_by_key(th_labels_copy.begin(), th_labels_copy.end(), thrust::make_constant_iterator(1),
		th_unique_keys.begin(), th_reduced_counts.begin());

	// After this the arrays look like this:
	// th_unique_keys:     0 1 2 3 4 7
	// th_reduced_counts:  2 2 3 2 2 1

	// Global d_counts array might be larger. In our example clusters 5 and 6 
	// are empty. So scatter takes th_reduced_counts[i] value and puts under
	// dev_counts[th_unique_keys[i]].
	thrust::device_ptr<int> dev_counts(data->d_counts);
	thrust::scatter(th_reduced_counts.begin(), end_counts.second, th_unique_keys.begin(), dev_counts);

	// So after it d_counts will look like this:
	// d_counts: 2 2 3 2 2 0 0 1 

	thrust::device_ptr<float> dev_points(data->d_points);
	thrust::device_ptr<double> dev_sums(data->d_sums);

	// Since our data is in SoA format, X, Y, Z, ... coordinates are stored in
	// separate blocks. We process each dimension independently.
	for (int j = 0; j < D; j++)
	{
		// Actual coordiantes in dev_points are still in unsorted original order.
		// th_indices tells us how points were reordered during sorting.
		// We use it to gather coords so they align with sorted labels.
		thrust::gather(th_indices.begin(), th_indices.end(), dev_points + (j * N), th_coords_gathered.begin());

		// I.e. after this
		// th_coords_gathered: x8 x10 x2 x5 x0 x3 x7 ... x11

		// So then we use reduce_by_key again and sum all x's according to
		// th_labels_copy keys.
		auto end_sums = thrust::reduce_by_key(th_labels_copy.begin(), th_labels_copy.end(), th_coords_gathered.begin(),                  
			th_unique_keys.begin(), th_reduced_sums.begin());

		// Now we have th_reduced_sums: (x8 + x10), (x2 + x5), (x0 + x3 + x7), ... , x11

		// Finally we scatter the calculated sums into the global 'd_sums' array
		// at the correct indices for the current dimension 'j'
		thrust::scatter(th_reduced_sums.begin(), end_sums.second, th_unique_keys.begin(), dev_sums + (j * K));
	}
	CHECK_CUDA(cudaDeviceSynchronize());

	// Now as d_sums and d_counts are ready, we can finally calculate 
	// new centroids coords, launching Update Kernel.
	int blocks_update = (K * D + BLOCK_SIZE - 1) / BLOCK_SIZE;
	calculate_centroids_and_delta << <blocks_update, BLOCK_SIZE >> > (data->d_centroids, data->d_sums, data->d_counts,
		K, D);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	// Copy the changes counter back to CPU. (As in GPU1, the rest will be copied
	// only after the algo is finished totally).
	CHECK_CUDA(cudaMemcpy(points_changed, data->d_changes_count, sizeof(int), cudaMemcpyDeviceToHost));

	(*iteration_number)++;
}