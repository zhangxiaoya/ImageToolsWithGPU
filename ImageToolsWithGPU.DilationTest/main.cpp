#include <highgui/highgui.hpp>
#include <core/core.hpp>
#include <iostream>
#include <core/cuda.hpp>
#include "opencv2/cudafilters.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include "kernel.cuh"
#include "../ImageToolsWithGPU.MatToArray/Models/LogLevel.hpp"
#include "../ImageToolsWithGPU.MatToArray/GlobalConsistantConfigure.h"
#include <vector_functions.h>

#define IMAX(a,b) (a > b) ? a : b;
#define IMIN(a,b) (a < b) ? a : b;

inline unsigned char ucMax(unsigned char a, unsigned char b)
{
	return (a > b) ? a : b;
}

inline unsigned char ucMin(unsigned char a, unsigned char b)
{
	return (a > b) ? b : a;
}

inline void DilationCPU(cv::Mat& srcFrame, cv::Mat& dstFrame, int kernelSize)
{
	auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	//	morphologyEx(frameNeedDetect, frameAfterMaxFilter, CV_MOP_OPEN, kernel);
	dilate(srcFrame, dstFrame, kernel);
}

inline bool Check(unsigned char* resultOnCPU, unsigned char* resultOnGPU, int width, int height)
{
	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			if (resultOnCPU[r * width + c] != resultOnGPU[r * width + c])
			{
				std::ostringstream oss;
				oss << "Expected: " << static_cast<int>(resultOnCPU[r * width + c]) << ", actual: " << static_cast<int>(resultOnGPU[r * width + c]) << ", on: " << r << ", " << c << std::endl;
				auto errorMsg = oss.str();
				logPrinter.PrintLogs(errorMsg, LogLevel::Error);
				return false;
			}
		}
	}
	return true;
}

bool Check(cv::Mat frameOnCPU, cv::Mat frameOnGPU)
{
	for(auto r = 0;r < frameOnCPU.rows;++r)
	{
		auto srcptr = frameOnCPU.ptr<uchar>(r);
		auto dstptr = frameOnGPU.ptr<uchar>(r);
		for(auto c = 0;c < frameOnCPU.cols;++c)
		{
			if(srcptr[c] != dstptr[c])
			{
				std::ostringstream oss;
				oss << "Expected: " << static_cast<int>(srcptr[c]) << ", actual: " << static_cast<int>(dstptr[c]) << ", on: " << r << ", " << c << std::endl;
				auto errorMsg = oss.str();
				logPrinter.PrintLogs(errorMsg, LogLevel::Error);
				return false;
			}
		}
	}
	return true;
}

inline void LevelDiscretizationOnCPU(cv::Mat& frame, int discretizationScale)
{
	for(auto r = 0;r < frame.rows;++r)
	{
		auto ptr = frame.ptr<unsigned char>(r);
		for(auto c = 0;c < frame.cols;++c)
		{
			ptr[c] = static_cast<unsigned char>(static_cast<int>(ptr[c]) / discretizationScale * discretizationScale);
		}
	}
}

int main(int argc, char* argv[])
{
	auto imageFileName = "C:\\D\\Cabins\\Projects\\Project1\\OriginalFrames\\second\\Frames\\ir_file_20170531_1000m_1\\Frame_00000000.png";

	auto img = cv::imread(imageFileName);
	if (img.empty())
	{
		std::cout << "Read image file failed!" << std::endl;
		system("Pause");
		return -1;
	}
	imshow("Original Image", img);
	cv::waitKey(0);

	cv::Mat grayImg;
	if (img.channels() == 3)
	{
		cvtColor(img, grayImg, CV_RGB2GRAY);
	}
	else
	{
		grayImg = img;
	}

	auto dilateRadius = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilateRadius + 1, 2 * dilateRadius + 1));

	cv::cuda::GpuMat imgOnDevice;
	imgOnDevice.upload(grayImg);

	cv::Ptr<cv::cuda::Filter> dilateFilterOnDevice = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, imgOnDevice.type(), element);
	dilateFilterOnDevice->apply(imgOnDevice, imgOnDevice);

	cv::Mat llimg;
	imgOnDevice.download(llimg);
	imshow("After Dilation on GPU", llimg);

	cv::Mat grayImgDilation;
	DilationCPU(grayImg, grayImgDilation, 3);

	if(Check(grayImgDilation, llimg))
		std::cout << "Dilation on CPU and GPU is same" <<std::endl;
	else
		std::cout << "Dilation on CPU and GPU is NOT same" << std::endl;

	if(imgOnDevice.isContinuous())
		std::cout << "Is Continus" <<std::endl;
	else
		std::cout << "Not Continus" <<std::endl;

//	auto data = imgOnDevice.data;
//	LevelDiscretizationOnGPU(data, 320, 256, 15);

	cv::cuda::GpuMat imgOnDeviceStep1;
	cv::cuda::GpuMat imgOnDeviceStep2;

	cv::cuda::divide(imgOnDevice, 15, imgOnDeviceStep1);
	cv::cuda::multiply(imgOnDeviceStep1, 15, imgOnDeviceStep2);

	cv::Mat ldImg;
	imgOnDeviceStep2.download(ldImg);

	LevelDiscretizationOnCPU(grayImgDilation, 15);

	if(Check(grayImgDilation, ldImg))
		std::cout << "Discreatized On CPU and GPU is same!" << std::endl;
	else
		std::cout << "Discreatized On CPU and GPU is NOT same!" << std::endl;

	for(auto i = 0; i< 10;++i)
	{
		cv::Mat ii = ldImg == i * 15;

		cv::cuda::GpuMat mask;
		mask.create(ii.rows, ii.cols, CV_8UC1);
		cv::cuda::GpuMat components;
		cv::cuda::connectivityMask(cv::cuda::GpuMat(ii), mask, cv::Scalar::all(0), cv::Scalar::all(2));

		cv::cuda::labelComponents(mask, components);

		int a = 0;
		int b = a;
	}

	imgOnDevice.release();
	imgOnDeviceStep1.release();
	imgOnDeviceStep2.release();
	imshow("After Level Discretizated on GPU", ldImg);
	cv::waitKey(0);

	cv::destroyAllWindows();
	system("Pause");
	return 0;
}
