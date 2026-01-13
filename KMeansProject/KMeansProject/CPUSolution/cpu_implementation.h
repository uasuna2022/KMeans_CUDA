#pragma once
#include <vector>

void run_k_means_algo(int max_iterations, float eps, const std::vector<float>& h_points,
	std::vector<float>& h_centroids, std::vector<int>& h_labels, int k, int n, int d);

