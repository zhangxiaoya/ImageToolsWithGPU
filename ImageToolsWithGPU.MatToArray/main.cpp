#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

#include "cuda_runtime.h"
#include "kernel.h"
#include "LogPrinter/LogPrinter.hpp"

const unsigned WIDTH = 320;
const unsigned HEIGHT = 256;
const unsigned BYTESIZE = 2;
const unsigned WHOLESIZE = WIDTH * HEIGHT;

LogPrinter logPrinter;

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
 * \brief init space on host, if init space for all image frame success, then return true;
 *        else roll back all operations and return false.
 * \param frameCount count of all frame in one file
 */
bool InitSpaceOnHost(unsigned frameCount)
{
	for (auto i = 0; i < frameCount; ++i)
	{
		auto cuda_error = cudaMallocHost(&allImageDataOnHost[i], WHOLESIZE);
		if (cuda_error != cudaSuccess)
		{
			logPrinter.PrintLogs("Init space on host failed! Starting roll back ...", LogLevel::Error);

			for (auto j = i - 1; j >= 0; j--)
				cudaFreeHost(allImageDataOnHost[j]);
			logPrinter.PrintLogs("Roll back done!", Info);
			return false;
		}
	}
	return true;
}

/**
 * \brief init space on device, if init space for all image frame success, then return true;
 *        else roll back all operations and return false.
 * \param frameCount the count of all frame in one file
 */
bool InitSpaceOnDevice(unsigned frameCount)
{
	for(auto i =0; i< frameCount;++i)
	{
		auto cuda_error = cudaMalloc(&allImageDataOnDevice[i], WHOLESIZE);
		if(cudaSuccess != cuda_error)
		{
			logPrinter.PrintLogs("Init space on device failed! Starting roll back ...", LogLevel::Error);

			for (auto j = i - 1; j >= 0; j--)
				cudaFree(allImageDataOnDevice[j]);
			logPrinter.PrintLogs("Roll back done!", Info);
			return false;
		}
	}
	return true;
}

bool LoadBinaryFIleToHostMemory()
{
	// create one binary file reader
	std::ifstream fin;
	OpenBinaryFile(fin);

	// temparory array
	auto originalPerFramePixelArray = new uint16_t[WIDTH * HEIGHT];
	auto iterationText = new char[200];

	if(fin.is_open())
	{
		// counting frame and init space on host and device respectly
		logPrinter.PrintLogs("Start binary file reading ...", LogLevel::Info);
		auto frameCount = GetFrameCount(fin);
		logPrinter.PrintLogs("The image count in this binary file is ", LogLevel::Info, frameCount);

		logPrinter.PrintLogs("Start init space on host ...", LogLevel::Info);
		auto init_space_on_host = InitSpaceOnHost(frameCount);
		auto init_space_on_device = InitSpaceOnDevice(frameCount);
		if(init_space_on_device && init_space_on_host)
		{
			logPrinter.PrintLogs("Init space on host and device done!", LogLevel::Info);
		}
		else
		{
			return false;
		}

		 // init some variables
		auto row = 0;              // current row index
		auto col = 0;              // current col index
		auto byteIndex = 2;        // current byte index
		auto frameIndex = 0;       // current frame index
		auto pixelIndex = 0;       // current pixel index

		uint8_t highPart = fin.get();
		uint8_t lowPart = fin.get();

		// main loop to read and load binary file per frame
		while (true)
		{
			// check if is the end of binary file
			if (!fin)
				break;

			// per frame
			while (byteIndex - 2 < WIDTH * HEIGHT * BYTESIZE)
			{
				// take 16-bit space per pixel
				uint16_t perPixel;
				ConstitudePixel(highPart, lowPart, perPixel);

				// but we only need only low part of one pixel (temparory)
				originalPerFramePixelArray[pixelIndex] = perPixel;
				allImageDataOnHost[frameIndex][pixelIndex] = lowPart;

				// update these variables
				ChangeRows(row, col);
				highPart = fin.get();
				lowPart = fin.get();
				byteIndex += 2;
				pixelIndex++;
			}
			sprintf_s(iterationText, 200, "Current frame index is %04d", frameIndex);
			logPrinter.PrintLogs(iterationText, LogLevel::Info);

			// prepare for next frame
			frameIndex++;
			row = 0;
			col = 0;
			byteIndex = 2;
			pixelIndex = 0;
		}

		// clean up temparory array
		if (originalPerFramePixelArray != nullptr)
		{
			delete[] originalPerFramePixelArray;
			originalPerFramePixelArray = nullptr;
		}
		if(iterationText != nullptr)
		{
			delete[] iterationText;
			iterationText = nullptr;
		}
	}
	else
	{
		// if open binary file failed!
		logPrinter.PrintLogs("Open binary file failed, please check file path!", LogLevel::Error);
		if (originalPerFramePixelArray != nullptr)
		{
			delete[] originalPerFramePixelArray;
			originalPerFramePixelArray = nullptr;
		}
		return false;
	}
	return true;
}

inline bool cudaDeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		logPrinter.PrintLogs("CUDA error: no devices supporting CUDA.", LogLevel::Error);
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

	system("Pause");
	return 0;
}
