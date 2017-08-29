#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

const unsigned WIDTH = 320;
const unsigned HEIGHT = 256;
const unsigned BYTESIZE = 2;

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

void GetFileLength(std::ifstream& fin)
{
	fin.seekg(0, std::ios::end);
	auto len = fin.tellg();
	std::cout << "Length" << len << std::endl;
	std::cout << "Frame Number:" << len * 1.0 / (WIDTH * HEIGHT * 2) << std::endl;
	fin.seekg(0, std::ios::beg);
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

bool LoadBinaryFIleToHostMemory(int& status)
{
	std::string binaryFileFullName = "C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1.bin";

	std::ifstream fin(binaryFileFullName, std::fstream::binary | std::fstream::in);

	auto originalPerFramePixelArray = new uint16_t[WIDTH * HEIGHT];
	auto afterLineartransformPerFramePixelArray = new uint8_t[WIDTH * HEIGHT];
	auto lowpartPerFramePixelArray = new uint8_t[WIDTH * HEIGHT];

	if(fin.is_open())
	{
		GetFileLength(fin);

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

				ChangeRows(row, col);
				highPart = fin.get();
				lowPart = fin.get();
				byteIndex += 2;
				pixelIndex++;
			}

			uint16_t maxValue;
			uint16_t minValue;
			auto diff = GetMaxDiff(originalPerFramePixelArray, maxValue, minValue);

			Lineartransform(originalPerFramePixelArray, afterLineartransformPerFramePixelArray, minValue, diff);

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
		if (afterLineartransformPerFramePixelArray != nullptr)
		{
			delete[] afterLineartransformPerFramePixelArray;
			afterLineartransformPerFramePixelArray = nullptr;
		}
		if(lowpartPerFramePixelArray != nullptr)
		{
			delete[] lowpartPerFramePixelArray;
			lowpartPerFramePixelArray = nullptr;
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
		if (afterLineartransformPerFramePixelArray != nullptr)
		{
			delete[] afterLineartransformPerFramePixelArray;
			afterLineartransformPerFramePixelArray = nullptr;
		}
		if (lowpartPerFramePixelArray != nullptr)
		{
			delete[] lowpartPerFramePixelArray;
			lowpartPerFramePixelArray = nullptr;
		}
		system("PAUSE");
		status = -1;
		return true;
	}
	return false;
}

int main(int argc, char* argv[])
{
	int loadFileStatus;
	if (LoadBinaryFIleToHostMemory(loadFileStatus))
		return loadFileStatus;

	system("PAUSE");
	return 0;
}
