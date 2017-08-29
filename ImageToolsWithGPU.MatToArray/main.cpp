#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

#include "cuda_runtime.h"
#include "kernel.h"

const unsigned WIDTH = 320;
const unsigned HEIGHT = 256;
const unsigned BYTESIZE = 2;
const unsigned WHOLESIZE = WIDTH * HEIGHT;

uint8_t* allImageDataOnHost[WHOLESIZE];
uint8_t* allImageDataOnDevice[WHOLESIZE];

void ConstitudePixel(uint8_t highPart, uint8_t lowPart, uint16_t& perPixel)
{
	perPixel = static_cast<uint16_t>(highPart);
	perPixel = perPixel << 8;
	perPixel |= lowPart;
}

void ChangeRows(int& row, int& col)
{
	col++;
	if (col == WIDTH)
	{
		col = 0;
		row++;
	}
}

uint16_t GetMaxDiff(uint16_t pixelArray[], uint16_t& maxValue, uint16_t& minValue)
{
	maxValue = 0;
	minValue = 1 << 15;
	for (auto pixelIndex = 0; pixelIndex < HEIGHT; ++pixelIndex)
	{
		if (pixelArray[pixelIndex] <= 255)
			continue;
		if (pixelArray[pixelIndex] < minValue)
			minValue = pixelArray[pixelIndex];
		if (pixelArray[pixelIndex] > maxValue)
			maxValue = pixelArray[pixelIndex];
	}
	return maxValue - minValue + 1;
}

unsigned GetFrameCount(std::ifstream& fin)
{
	fin.seekg(0, std::ios::end);
	auto len = fin.tellg();
	auto frameCount = len * 1.0 / (WIDTH * HEIGHT * 2);
	fin.seekg(0, std::ios::beg);
	return frameCount;
}

void Lineartransform(const unsigned short* originalPerFramePixelArray, unsigned char* afterLineartransformPerFramePixelArray, const uint16_t minValue, const uint16_t diff)
{
	auto sacle = static_cast<double>(diff) / 255;
	for (auto pixelIndex = 0; pixelIndex < WIDTH; ++pixelIndex)
	{
		auto value = double(originalPerFramePixelArray[pixelIndex] - minValue + 1) / sacle;
		if (value > 255)
			value = 255;
		afterLineartransformPerFramePixelArray[pixelIndex] = static_cast<uint8_t>(value);
	}
}

void OpenBinaryFile(std::ifstream& fin)
{
	std::string binaryFileFullName = "C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin";

	fin = std::ifstream(binaryFileFullName, std::fstream::binary | std::fstream::in);
}

/**
 * \brief init space on host
 * \param frameCount count of all frame in one file
 */
void InitSpaceOnHost(unsigned frameCount)
{
	for (auto i = 0; i < frameCount; ++i)
	{
		cudaMallocHost(&allImageDataOnHost[i], WHOLESIZE);
	}
}

/**
 * \brief init space on device
 * \param frameCount the count of all frame in one file
 */
void InitSpaceOnDevice(unsigned frameCount)
{
	for(auto i =0; i< frameCount;++i)
	{
		cudaMalloc(&allImageDataOnDevice[i], WHOLESIZE);
	}
}

bool LoadBinaryFIleToHostMemory()
{
	std::ifstream fin;
	OpenBinaryFile(fin);

	auto originalPerFramePixelArray = new uint16_t[WIDTH * HEIGHT];
	if(fin.is_open())
	{
		auto frameCount = GetFrameCount(fin);
		InitSpaceOnHost(frameCount);
		InitSpaceOnDevice(frameCount);

		auto row = 0;
		auto col = 0;
		auto byteIndex = 2;
		auto frameIndex = 0;
		auto pixelIndex = 0;

		uint8_t highPart = fin.get();
		uint8_t lowPart = fin.get();

		while (true)
		{
			if (!fin)
				break;

			while (byteIndex - 2 < WIDTH * HEIGHT * BYTESIZE)
			{
				uint16_t perPixel;
				ConstitudePixel(highPart, lowPart, perPixel);

				originalPerFramePixelArray[pixelIndex] = perPixel;
				allImageDataOnHost[frameIndex][pixelIndex] = lowPart;

				ChangeRows(row, col);
				highPart = fin.get();
				lowPart = fin.get();
				byteIndex += 2;
				pixelIndex++;
			}

			std::cout << "Frame Index ==> " << std::setw(4) << frameIndex << std::endl;

			frameIndex++;
			row = 0;
			col = 0;
			byteIndex = 2;
			pixelIndex = 0;
		}

		if (originalPerFramePixelArray != nullptr)
		{
			delete[] originalPerFramePixelArray;
			originalPerFramePixelArray = nullptr;
		}
	}
	else
	{
		std::cout << "Open binary file failed, please check file path!"<<std::endl;
		if (originalPerFramePixelArray != nullptr)
		{
			delete[] originalPerFramePixelArray;
			originalPerFramePixelArray = nullptr;
		}
		return true;
	}
	return false;
}

inline bool cudaDeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(0);
	return true;
}

int main(int argc, char** argv)
{
	if(cudaDeviceInit(argc, const_cast<const char **>(argv)))
	{
		if (LoadBinaryFIleToHostMemory())
		{
		}
	}

	return 0;
}
