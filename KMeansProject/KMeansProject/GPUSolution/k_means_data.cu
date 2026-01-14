#include "k_means_data.h"

void KMeansData::allocate_memory()
{
	CHECK_CUDA(cudaMalloc((void**)&d_points, n * d * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void**)&d_centroids, k * d * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void**)&d_labels, n * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void**)&d_sums, k * d * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void**)&d_counts, k * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void**)&d_deltas, k * sizeof(float)));
	CHECK_CUDA(cudaMalloc((void**)&d_changes_count, sizeof(int)));
}


KMeansData::KMeansData(int n_input, int k_input, int d_input) :
	n(n_input), k(k_input), d(d_input), d_points(nullptr), d_centroids(nullptr), 
	d_labels(nullptr), d_sums(nullptr), d_counts(nullptr), d_deltas(nullptr)
{
	allocate_memory();
}

void KMeansData::free_memory()
{
	if (d_points)
		cudaFree(d_points);
	if (d_centroids)
		cudaFree(d_centroids);
	if (d_labels)
		cudaFree(d_labels);
	if (d_sums)
		cudaFree(d_sums);
	if (d_counts)
		cudaFree(d_counts);
	if (d_deltas)
		cudaFree(d_deltas);
	if (d_changes_count)
		cudaFree(d_changes_count);
}

KMeansData::~KMeansData()
{
	free_memory();
}

void KMeansData::fill_gpu_data(const std::vector<float>& h_points_soa, const std::vector<float>& h_centroids_soa)
{
	CHECK_CUDA(cudaMemcpy(d_points, h_points_soa.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids_soa.data(), k * d * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemset(d_labels, 0, n * sizeof(int)));
	CHECK_CUDA(cudaMemset(d_counts, 0, k * sizeof(int)));
	CHECK_CUDA(cudaMemset(d_sums, 0.0F, k * d * sizeof(float)));
	CHECK_CUDA(cudaMemset(d_deltas, 0.0F, k * sizeof(float)));
	CHECK_CUDA(cudaMemset(d_changes_count, 0, sizeof(int)));
}
