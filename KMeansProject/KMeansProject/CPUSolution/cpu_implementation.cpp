#include "cpu_implementation.h"
#include <iostream>

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

void run_k_means_algo(int max_iterations, float eps, const std::vector<float>& h_points,
	std::vector<float>& h_centroids, std::vector<int>& h_labels, int k, int n, int d)
{
	int iteration_number = 1;
	std::vector<float> centroids_sum;
	std::vector<int> cluster_counter;
	std::vector<float> old_centroids;
	centroids_sum.resize(k * d);
	cluster_counter.resize(k);
	old_centroids.resize(k * d);

	while (iteration_number <= max_iterations)
	{
		old_centroids = h_centroids;
		std::fill(centroids_sum.begin(), centroids_sum.end(), 0.0F);
		std::fill(cluster_counter.begin(), cluster_counter.end(), 0);

		int changed_cluster_points = 0;
		for (int i = 0; i < n; i++)
		{
			int best_cluster = -1;
			float best_distance = std::numeric_limits<float>::max();

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

		if (changed_cluster_points == 0)
		{
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

		float total_shift = 0.0F;
		for (int i = 0; i < k; i++)
		{
			total_shift += std::sqrt(calculate_distance_squared(old_centroids, h_centroids, d, i, i));
		}
		
		if (total_shift < eps)
		{
			std::cout << "Algorithm has been stopped as centroids have become stable (delta < epsilon) in iteration nr. " << 
				iteration_number << std::endl;
			break;
		}

		iteration_number++;
	}

	if (iteration_number > max_iterations)
		std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;
}