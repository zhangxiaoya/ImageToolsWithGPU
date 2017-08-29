#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

cudaError_t CopyDataToDevice(unsigned frameCount, unsigned char** allImageDataOnHost, unsigned char** allImageDataOnDevice, unsigned width, unsigned height)
{
	auto cudaStatus = cudaSuccess;
	for (auto i = 0; i < frameCount; ++i)
	{
		cudaStatus = cudaMemcpy(allImageDataOnDevice[i], allImageDataOnHost[i], width * height, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			return cudaStatus;
		}
	}
	return cudaStatus;
}