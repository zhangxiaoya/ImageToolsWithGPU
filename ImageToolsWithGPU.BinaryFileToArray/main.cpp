#include <iostream>
#include <string>
#include <fstream>

/*
 *  this is only for one binary file
 */

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

uint16_t OpenBinaryFile(uint16_t pixelValues[], uint16_t& minVlaue, uint16_t& maxValue)
{
	minVlaue = pixelValues[0];
	maxValue = 0;

	for (auto i = 0; i < WIDTH * HEIGHT; ++i)
	{
		if (pixelValues[i] < static_cast<uint16_t>(255))
			continue;
		if (maxValue < pixelValues[i])
			maxValue = pixelValues[i];
		if (minVlaue > pixelValues[i])
			minVlaue = pixelValues[i];
	}
	return maxValue - minVlaue + 1;
}

int main(int argc, char* argv[])
{
	std::string binaryFileName = "D:\\Bag\\Code_VS15\\Data\\6km\\target_in_6km_659.dat";
	std::ifstream fin(binaryFileName, std::fstream::in | std::fstream::binary);

	if (fin.is_open())
	{
		auto pixelArray = new uint16_t[WIDTH * HEIGHT];
		auto newPixelArray = new uint8_t[WIDTH * HEIGHT];

		auto row = 0;
		auto col = 0;
		auto byteIndex = 0;
		auto pixelIndex = 0;

		while (byteIndex < WIDTH * HEIGHT * BYTESIZE)
		{
			uint8_t highPart = fin.get();
			uint8_t lowPart = fin.get();

			uint16_t perPixel;
			ConstitudePixel(highPart, lowPart, perPixel);
			pixelArray[pixelIndex] = perPixel;

			byteIndex += 2;
			ChangeRows(row, col);
		}

		uint16_t maxValue;
		uint16_t minVlaue;

		auto maxDiff = OpenBinaryFile(pixelArray, minVlaue, maxValue);

		auto scale = static_cast<double>(maxDiff) / 256;
		for (auto i = 0; i < WIDTH * HEIGHT;++i)
		{
			if (pixelArray[i] < minVlaue)
				newPixelArray[i] = 0;
			auto value = floor((pixelArray[i] - minVlaue + 1) / scale);
			newPixelArray[i] = static_cast<uint8_t>(value);
		}

		for (auto i = 0; i < WIDTH * HEIGHT;++i)
		{
			std::cout << static_cast<int>(newPixelArray[i]) << " ";
			if (i % WIDTH == 0)
				std::cout << std::endl;;
		}

		// clean up pixel array
		if(pixelArray != nullptr)
		{
			delete[] pixelArray;
			pixelArray = nullptr;
		}
		if(newPixelArray != nullptr)
		{
			delete[] newPixelArray;
			newPixelArray = nullptr;
		}
	}
	else
	{
		std::cout << "Open Binary File Failed" << std::endl;
		system("Pause");
		return -1;
	}
	system("Pause");
	return 0;
}
