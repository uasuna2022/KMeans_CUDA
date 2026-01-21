#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "CPUSolution/cpu_implementation.h"
#include "GPUSolution/k_means_data.h"
#include "GPUSolution/gpu1_kernel_functions.cuh"
#include "GPUSolution/gpu2_kernel_functions.cuh"

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

void cpu_transform_aos_to_soa(int n, int d, const std::vector<float>& aos, std::vector<float>& soa)
{
	soa.resize(n * d);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			soa[j * n + i] = aos[i * d + j];
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

	/* testing features
	std::vector<float> h_points_copy = h_points;
	std::vector<float> h_centroids_copy = h_centroids;

	auto start_time_1 = std::chrono::high_resolution_clock::now();
	run_k_means_algo(500, 0.0001F, h_points, h_centroids, h_labels, k, n, d);
	auto end_time_1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_1 = end_time_1 - start_time_1;
	
	KMeansData data_gpu1(n, k, d);
	std::vector<float> h_points_soa;
	std::vector<float> h_centroids_soa;
	cpu_transform_aos_to_soa(n, d, h_points_copy, h_points_soa);
	cpu_transform_aos_to_soa(k, d, h_centroids_copy, h_centroids_soa);

	data_gpu1.fill_gpu_data(h_points_soa, h_centroids_soa);

	int max_iterations = 500;
	int iteration_number = 0;
	int points_changed = 0;
	float eps = 0.0001F;
	float delta = 0.0F;

	auto start_time_2 = std::chrono::high_resolution_clock::now();
	while (iteration_number <= max_iterations)
	{
		auto start_time_it = std::chrono::high_resolution_clock::now();
		make_iteration(&data_gpu1, &iteration_number, &delta, &points_changed);
		auto end_time_it = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_it = end_time_it - start_time_it;
		std::cout << "Iteration " << iteration_number << ": " << elapsed_it.count() << " | delta = " << delta
			<< " | Points changed cluster = " << points_changed << std::endl;

		if (delta < eps)
		{
			std::cout << "Algorithm has been stopped as as centroids have become stable (delta < epsilon) in iteration nr. " <<
				iteration_number << std::endl;
			break;
		}
		if (points_changed == 0)
		{
			std::cout << "Algorithm has been stopped as no points have changed their cluster in iteration nr. " <<
				iteration_number << std::endl;
			break;
		}
	}
	if (iteration_number > max_iterations)
	{
		std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;
	}
	auto end_time_2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_2 = end_time_2 - start_time_2;

	std::cout << "CPU time: " << elapsed_1.count() << std::endl << "GPU1 time: " << elapsed_2.count();

	return EXIT_SUCCESS;
	*/

	auto start_time = std::chrono::high_resolution_clock::now();
	if (version == "cpu")
	{
		run_k_means_algo(500, 0.0001F, h_points, h_centroids, h_labels, k, n, d);
	}
	else if (version == "gpu1")
	{
		KMeansData data(n, k, d);

		std::vector<float> h_points_soa;
		std::vector<float> h_centroids_soa;
		cpu_transform_aos_to_soa(n, d, h_points, h_points_soa);
		cpu_transform_aos_to_soa(k, d, h_centroids, h_centroids_soa);

		data.fill_gpu_data(h_points_soa, h_centroids_soa);

		int max_iterations = 500;
		int iteration_number = 0;
		int points_changed = 0;
		float eps = 0.0001F;
		float delta = 0.0F;

		while (iteration_number <= max_iterations)
		{
			auto start_time_it = std::chrono::high_resolution_clock::now();
			make_iteration(&data, &iteration_number, &delta, &points_changed);
			auto end_time_it = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_it = end_time_it - start_time_it;
			std::cout << "Iteration " << iteration_number << ": " << elapsed_it.count() << " | delta = " << delta  
				<< " | Points changed cluster = " << points_changed << std::endl;

			if (delta < eps)
			{
				std::cout << "Algorithm has been stopped as as centroids have become stable (delta < epsilon) in iteration nr. " <<
					iteration_number << std::endl;
				break;
			}
			if (points_changed == 0)
			{
				std::cout << "Algorithm has been stopped as no points have changed their cluster in iteration nr. " <<
					iteration_number << std::endl;
				break;
			}
		}
		if (iteration_number > max_iterations)
		{
			std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;
		}
	}
	else 
	{
		KMeansData data(n, k, d);

		std::vector<float> h_points_soa;
		std::vector<float> h_centroids_soa;
		cpu_transform_aos_to_soa(n, d, h_points, h_points_soa);
		cpu_transform_aos_to_soa(k, d, h_centroids, h_centroids_soa);

		data.fill_gpu_data(h_points_soa, h_centroids_soa);

		int max_iterations = 500;
		int iteration_number = 0;
		int points_changed = 0;
		float eps = 0.0001F;
		float delta = 0.0F;

		while (iteration_number <= max_iterations)
		{
			auto start_time_it = std::chrono::high_resolution_clock::now();
			make_iteration_2(&data, &iteration_number, &delta, &points_changed);
			auto end_time_it = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_it = end_time_it - start_time_it;
			std::cout << "Iteration " << iteration_number << ": " << elapsed_it.count() << " | delta = " << delta
				<< " | Points changed cluster = " << points_changed << std::endl;

			if (delta < eps)
			{
				std::cout << "Algorithm has been stopped as as centroids have become stable (delta < epsilon) in iteration nr. " <<
					iteration_number << std::endl;
				break;
			}
			if (points_changed == 0)
			{
				std::cout << "Algorithm has been stopped as no points have changed their cluster in iteration nr. " <<
					iteration_number << std::endl;
				break;
			}
		}
		if (iteration_number > max_iterations)
		{
			std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;
		}
	} 

	
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	std::cout << "Time elapsed: " << elapsed.count() << std::endl;
	return 0;
}