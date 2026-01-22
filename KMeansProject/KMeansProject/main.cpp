#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "CPUSolution/cpu_implementation.h"
#include "GPUSolution/k_means_data.h"
#include "GPUSolution/gpu1_kernel_functions.cuh"
#include "GPUSolution/gpu2_kernel_functions.cuh"
#include "file_io.h"

#define MAX_ITERATIONS 100

/*
void usage(int argc, char** argv)
{
	std::cout << "USAGE: " << argv[0] << " <version> k n d\n";
	std::cout << "\t<version> - cpu/gpu1/gpu2\n";
	std::cout << "\tk - number of clusters (2 <= k <= 10)\n";
	std::cout << "\tn - number of points (1000 <= n <= 10000000)\n";
	std::cout << "\td - dimension (1 <= d <= 128)\n";
	exit(EXIT_FAILURE);
}
*/

void checkLoading(std::vector<float>& h_points, int n, int d, int k)
{
	std::cout << n << " " << d << " " << k << std::endl;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			std::cout << h_points[i * d + j] << " ";
		}
		std::cout << std::endl;
	}
}

/*
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
*/

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

void cpu_transform_soa_to_aos(int n, int d, const std::vector<float>& soa, std::vector<float>& aos)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			aos[i * d + j] = soa[j * n + i];
		}
	}
}

void print_usage(int argc, char** argv)
{
	std::cerr << "USAGE: " << argv[0] << " <data_format> <computation_method> <path_to_input_file> <path_to_output_file>\n";
	std::cerr << "\t<data_format>: txt | bin\n";
	std::cerr << "\t<computation_method>: cpu | gpu1 | gpu2\n";
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
	/*
	{
		// TESTING FEATURES
		int n;
		int k;
		int d;
		std::vector<float> h_points;
		const std::string filename = "./data/data.txt";
		if (!load_data(filename, "txt", n, d, k, h_points))
		{
			std::cout << "ERROR!" << std::endl;
			return EXIT_FAILURE;
		}
		else checkLoading(h_points, n, d, k);
		return EXIT_SUCCESS;
	}
	*/

	if (argc != 5)
		print_usage(argc, argv);

	std::string data_format = argv[1];
	std::string computation_method = argv[2];
	std::string input_file = argv[3];
	std::string output_file = argv[4];

	if (data_format != "txt" && data_format != "bin")
	{
		std::cerr << "Invalid input: data_format must be either 'txt' or 'bin'\n";
		print_usage(argc, argv);
	}
	if (computation_method != "cpu" && computation_method != "gpu1" && computation_method != "gpu2")
	{
		std::cerr << "Invalid input: computation_method must be either 'cpu' or 'gpu1' or 'gpu2'\n";
		print_usage(argc, argv);
	}

	auto start_total = std::chrono::high_resolution_clock::now();

	std::vector<float> h_points;
	std::vector<float> h_centroids;
	std::vector<int> h_labels;
	int n = 0;
	int d = 0;
	int k = 0;

	std::cout << "Reading data from input file to cpu..." << std::endl;

	auto start_read = std::chrono::high_resolution_clock::now();
	if (!load_data(input_file, data_format, n, d, k, h_points))
	{
		std::cerr << "Error: can't load data from input_file\n";
		return EXIT_FAILURE;
	}
	auto end_read = std::chrono::high_resolution_clock::now();
	double read_time = std::chrono::duration<double>(end_read - start_read).count();
	std::cout << "  Data format: " << data_format << std::endl;
	std::cout << "  Number of points: " << n << std::endl;
	std::cout << "  Dimension: " << d << std::endl;
	std::cout << "  Number of clusters: " << k << std::endl;
	std::cout << "Data read successfully. Elapsed time: " << read_time << "s" << std::endl << std::endl;

	h_labels.resize(n, -1);
	h_centroids.resize(k * d);

	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < d; j++)
		{
			h_centroids[i * d + j] = h_points[i * d + j];
		}
	}

	auto start_algo = std::chrono::high_resolution_clock::now();
	auto end_algo = std::chrono::high_resolution_clock::now();
	double algo_time;
	if (computation_method == "cpu")
	{
		int iterations_done = 0;
		start_algo = std::chrono::high_resolution_clock::now();
		run_k_means_algo(MAX_ITERATIONS, h_points, h_centroids, h_labels, k, n, d, &iterations_done);
		end_algo = std::chrono::high_resolution_clock::now();
		algo_time = std::chrono::duration<double>(end_algo - start_algo).count();
		std::cout << "Total computation time: " << std::fixed << std::setprecision(4) << algo_time << "s" << std::endl;
		std::cout << "Average time per iteration: " << algo_time / (double)iterations_done << "s" << std::endl;
	}
	else if (computation_method == "gpu1")
	{
		KMeansData data(n, k, d);
		std::vector<float> h_points_soa;
		std::vector<float> h_centroids_soa;

		std::cout << "Transforming data from AoS to SoA (for better GPU perfomance)...\n";
		auto start_transform = std::chrono::high_resolution_clock::now();
		cpu_transform_aos_to_soa(n, d, h_points, h_points_soa);
		cpu_transform_aos_to_soa(k, d, h_centroids, h_centroids_soa);
		auto end_transform = std::chrono::high_resolution_clock::now();
		std::cout << "Data transformed. Elapsed time: " << std::chrono::duration<double>(end_transform - start_transform).count() <<
			"s" << std::endl << std::endl;

		std::cout << "Copying data from CPU to GPU...\n";
		auto copy_start = std::chrono::high_resolution_clock::now();
		data.fill_gpu_data(h_points_soa, h_centroids_soa);
		auto copy_end = std::chrono::high_resolution_clock::now();
		std::cout << "Data copied. Elapsed time: " << std::chrono::duration<double>(copy_end - copy_start).count() << "s\n\n";

		int iteration_number = 1;
		int points_changed = 0;
		start_algo = std::chrono::high_resolution_clock::now();

		while (iteration_number <= MAX_ITERATIONS)
		{
			std::cout << "  Iteration " << std::setw(3) << iteration_number;
			auto start_it = std::chrono::high_resolution_clock::now();
			make_iteration(&data, &iteration_number, &points_changed);
			auto end_it = std::chrono::high_resolution_clock::now();
			double it_time = std::chrono::duration<double>(end_it - start_it).count();
			std::cout << " | time = " << std::fixed << std::showpoint << std::setprecision(4) <<
				it_time << "s | " << "changes = " << points_changed << std::endl;

			if (points_changed == 0)
			{
				std::cout << "Algorithm has been stopped as no points have changed their cluster in iteration nr. " <<
					iteration_number - 1 << std::endl;
				break;
			}
		}
		if (iteration_number > MAX_ITERATIONS)
			std::cout << "Algorithm has been stopped as maximum number of iterations happened" << std::endl;

		end_algo = std::chrono::high_resolution_clock::now();
		algo_time = std::chrono::duration<double>(end_algo - start_algo).count();
		std::cout << "Total computation time: " << std::fixed << std::setprecision(4) << algo_time << "s" << std::endl;
		std::cout << "Average time per iteration: " << algo_time / (double)(iteration_number - 1) << "s\n" << std::endl;


		std::cout << "Copying data from GPU to CPU...\n";
		copy_start = std::chrono::high_resolution_clock::now();
		CHECK_CUDA(cudaMemcpy(h_labels.data(), data.d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(h_centroids_soa.data(), data.d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost));
		copy_end = std::chrono::high_resolution_clock::now();
		std::cout << "Data copied. Elapsed time: " << std::chrono::duration<double>(copy_end - copy_start).count() << "s\n\n";

		std::cout << "Transforming data from SoA to AoS back...\n";
		start_transform = std::chrono::high_resolution_clock::now();
		cpu_transform_soa_to_aos(k, d, h_centroids_soa, h_centroids);
		end_transform = std::chrono::high_resolution_clock::now();
		std::cout << "Data transformed. Elapsed time: " << std::chrono::duration<double>(end_transform - start_transform).count() <<
			"s" << std::endl << std::endl;

	}
	else if (computation_method == "gpu2")
	{
		// TODO
	}

	std::cout << "Saving computed results to the output_file...\n";
	auto start_save = std::chrono::high_resolution_clock::now();
	if (!save_results(output_file, h_labels, h_centroids, n, k, d))
	{
		std::cerr << "Error: can't save results to output_file\n";
		return EXIT_FAILURE;
	}
	auto end_save = std::chrono::high_resolution_clock::now();
	double save_time = std::chrono::duration<double>(end_save - start_save).count();
	std::cout << "Results saved succesfully. Elapsed time: " << save_time << "s" << std::endl;

	auto end_total = std::chrono::high_resolution_clock::now();
	double total_time = std::chrono::duration<double>(end_total - start_total).count();
	std::cout << std::endl << "Total time: " << total_time << "s" << std::endl;

	return EXIT_SUCCESS;
}

	/*
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


	auto start_time = std::chrono::high_resolution_clock::now();
	if (version == "cpu")
	{
		run_k_means_algo(500, 0.001F, h_points, h_centroids, h_labels, k, n, d);
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
		float eps = 0.001F;
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
			if (points_changed <= (int)(0.0005 * n))
			{
				std::cout << "Algorithm has been stopped as less than 0.05% of points have changed their cluster in iteration nr. " <<
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
		float eps = 0.001F;
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
			if (points_changed <= (int)(0.0005 * n))
			{
				std::cout << "Algorithm has been stopped as less than 0.05% of points have changed their cluster in iteration nr. " <<
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
	*/
