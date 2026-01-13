#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "CPUSolution/cpu_implementation.h"

void usage(int argc, char** argv)
{
	std::cout << "USAGE: " << argv[0] << " <version> k n d\n";
	std::cout << "\t<version> - cpu/gpu1/gpu2\n";
	std::cout << "\tk - number of clusters (2 <= k <= 10)\n";
	std::cout << "\tn - number of points (1000 <= n <= 10000000)\n";
	std::cout << "\td - dimension (1 <= d <= 128)\n";
	exit(EXIT_FAILURE);
}

void initializeData(int k, int n, int d, std::vector<float>& h_points, std::vector<float>& h_centroids)
{
	std::mt19937 random_generator(time(NULL));
	std::uniform_real_distribution<float> distribution(0.0F, 1.0F);

	h_points.resize(n * d);
	h_centroids.resize(k * d);

	for (int i = 0; i < n * d; i++)
	{
		h_points[i] = distribution(random_generator);
	}

	std::vector<int> indices;
	for (int i = 0; i < n; i++)
	{
		indices.push_back(i);
	}
	std::shuffle(indices.begin(), indices.end(), random_generator);

	for (int i = 0; i < k; i++)
	{
		int random_index = indices[i];
		for (int j = 0; j < d; j++)
		{
			h_centroids[d * i + j] = h_points[random_index * d + j];
		}
	}
}

int main(int argc, char** argv)
{
	if (argc != 5)
		usage(argc, argv);

	std::string version = argv[1];
	if (version != "cpu" && version != "gpu1" && version != "gpu2")
		usage(argc, argv);

	int k = atoi(argv[2]);
	if (k > 10 || k < 2)
		usage(argc, argv);

	int n = atoi(argv[3]);
	if (n < 1000 || n > 10000000)
		usage(argc, argv);

	int d = atoi(argv[4]);
	if (d < 1 || d > 128)
		usage(argc, argv);

	
	std::vector<float> h_points;
	std::vector<float> h_centroids;
	std::vector<int> h_labels(n, -1);
	initializeData(k, n, d, h_points, h_centroids);

	if (version == "cpu")
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		run_k_means_algo(200, 0.0001F, h_points, h_centroids, h_labels, k, n, d);
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << "Time elapsed: " << elapsed.count() << std::endl;
	}
	else if (version == "gpu1")
	{
		// gpu1 logic
	}
	else {} // gpu2 logic
	
	return 0;
}