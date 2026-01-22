#include "cpu_implementation.h"
#include <iostream>
#include <cfloat>
#include <chrono>
#include <iomanip>

float calculate_distance_squared(const std::vector<float>& points, const std::vector<float>& centroids, int d, 
	int point_idx, int centroid_idx)
{
	float distance = 0.0F;
	for (int i = 0; i < d; i++)
	{
		float diff = points[point_idx * d + i] - centroids[centroid_idx * d + i];
		diff *= diff;
		distance += diff;
	}

	return distance;
}

void run_k_means_algo(int max_iterations, const std::vector<float>& h_points,
	std::vector<float>& h_centroids, std::vector<int>& h_labels, int k, int n, int d)
{
	int iteration_number = 1;
	std::vector<float> centroids_sum;
	std::vector<int> cluster_counter;
	centroids_sum.resize(k * d);
	cluster_counter.resize(k);

	while (iteration_number <= max_iterations)
	{
		auto it_start = std::chrono::high_resolution_clock::now();
		std::cout << "  Iteration " << std::setw(3) << iteration_number;
		std::fill(centroids_sum.begin(), centroids_sum.end(), 0.0F);
		std::fill(cluster_counter.begin(), cluster_counter.end(), 0);

		int changed_cluster_points = 0;
		for (int i = 0; i < n; i++)
		{
			int best_cluster = -1;
			float best_distance = FLT_MAX;

			for (int j = 0; j < k; j++)
			{
				float dist = calculate_distance_squared(h_points, h_centroids, d, i, j);
				if (dist < best_distance)
				{
					best_distance = dist;
					best_cluster = j;
				}
			}
			
			if (h_labels[i] != best_cluster)
			{
				h_labels[i] = best_cluster;
				changed_cluster_points++;
			}

			cluster_counter[best_cluster]++;
			for (int j = 0; j < d; j++)
			{
				centroids_sum[best_cluster * d + j] += h_points[i * d + j];
			}
		}

		auto it_end = std::chrono::high_resolution_clock::now();

		if (changed_cluster_points == 0)
		{
			
			std::cout << " | time = " << std::fixed << std::showpoint << std::setprecision(4) << 
				std::chrono::duration<double>(it_end - it_start).count() << "s | " <<
				"changes = " << changed_cluster_points << std::endl;
			std::cout << "Algorithm has been stopped as no points have changed their cluster in iteration nr. " << 
				iteration_number << std::endl;
			break;
		}

		for (int i = 0; i < k; i++)
		{
			if (cluster_counter[i] > 0)
			{
				for (int j = 0; j < d; j++)
				{
					h_centroids[i * d + j] = centroids_sum[i * d + j] / cluster_counter[i];
				}
			}
		}
		it_end = std::chrono::high_resolution_clock::now();
		double it_time = std::chrono::duration<double>(it_end - it_start).count();
		std::cout << " | time = " << std::fixed << std::showpoint << std::setprecision(4) <<
			std::chrono::duration<double>(it_end - it_start).count() << "s | " <<
			"changes = " << changed_cluster_points << std::endl;
		iteration_number++;
	}

	if (iteration_number > max_iterations)
		std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;

}