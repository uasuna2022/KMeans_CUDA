#pragma once
#include <vector>
#include <string>

bool load_data(const std::string& filename, const std::string& format,
	int& n, int& d, int& k, std::vector<float>& h_points);

bool save_results(const std::string& filename, const std::vector<int>& h_labels,
	const std::vector<float>& h_centroids, int n, int k, int d);