#pragma once 

#include "cuda_runtime.h"
#include "k_means_data.h"
#include <vector>

void make_iteration_2(KMeansData* data, int* iteration_number, int* points_changed);