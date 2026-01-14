#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "k_means_data.h"

void make_iteration(KMeansData* data, int* iteration_number, float* delta, int* points_changed);