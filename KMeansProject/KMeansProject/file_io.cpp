#include "file_io.h"

// Loads a data from text file to the appropriate variables.
bool load_text(const char* filename, int& n, int& d, int& k, std::vector<float>& h_points)
{
	FILE* f = fopen(filename, "r");
	if (!f)
		return false;

	if (fscanf(f, "%d %d %d", &n, &d, &k) != 3)
	{
		fclose(f);
		return false;
	}

	h_points.resize(n * d);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			float value;
			if (fscanf(f, "%f", &value) != 1)
			{
				fclose(f);
				return false;
			}
			h_points[i * d + j] = value;
		}
	}

	fclose(f);
	return true;
}

// Loads data from binary file to the appropriate variables.
bool load_binary(const char* filename, int& n, int& d, int& k, std::vector<float>& h_points)
{
	FILE* f = fopen(filename, "rb");
	if (!f)
		return false;

	if (fread(&n, 4, 1, f) != 1)
	{
		fclose(f);
		return false;
	}
	if (fread(&d, 4, 1, f) != 1)
	{
		fclose(f);
		return false;
	}
	if (fread(&k, 4, 1, f) != 1)
	{
		fclose(f);
		return false;
	}

	h_points.resize(n * d);

	std::vector<double> point_buffer(d);

	for (int i = 0; i < n; i++)
	{
		if (fread(point_buffer.data(), 8, d, f) != d)
		{
			fclose(f);
			return false;
		}

		for (int j = 0; j < d; j++)
		{
			h_points[i * d + j] = (float)point_buffer[j];
		}
	}

	fclose(f);
	return true;
}

// We call this function inside main.cpp.
bool load_data(const std::string& filename, const std::string& format,
	int& n, int& d, int& k, std::vector<float>& h_points)
{
	if (format == "txt")
		return load_text(filename.c_str(), n, d, k, h_points);
	else if (format == "bin")
		return load_binary(filename.c_str(), n, d, k, h_points);
	else return false;
}

// Saves results after algorithm flow to the output text file.
bool save_results(const std::string& filename, const std::vector<int>& h_labels,
	const std::vector<float>& h_centroids, int n, int k, int d)
{
	FILE* f = fopen(filename.c_str(), "w");
	if (!f)
		return false;

	for (int i = 0; i < k; i++)
	{
		fprintf(f, "  ");
		for (int j = 0; j < d; j++)
		{
			fprintf(f, "%.4f%s", h_centroids[i * d + j], j == (d - 1) ? "" : " ");
		}
		fprintf(f, "\n");
	}

	for (int i = 0; i < n; i++)
		fprintf(f, "  %d\n", h_labels[i]);
	

	fclose(f);
	return true;
}