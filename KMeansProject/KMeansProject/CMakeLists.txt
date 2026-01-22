cmake_minimum_required(VERSION 3.18)
project(KMeansGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 75 86 89)
endif()

set(SOURCES
    main.cpp
    CPUSolution/cpu_implementation.cpp
    GPUSolution/k_means_data.cu
    GPUSolution/gpu1_kernel_functions.cu
    GPUSolution/gpu2_kernel_functions.cu
    file_io.cpp
)

add_executable(KMeans ${SOURCES})

target_include_directories(KMeans PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(KMeans PRIVATE CUDA::cudart)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(KMeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)