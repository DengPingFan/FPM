#include <cufft.h>
#include <iostream>

//Cuda的fftShift变换
template <typename T>
__global__ void cufftShift_2D_kernel(T *data, int N)
{
	int sLine = N;
	int sSlice = N*N;

	//转换方程
	int sEq1 = (sSlice + sLine) / 2;
	int sEq2 = (sSlice - sLine) / 2;

	//线程坐标（2D）
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	//线程坐标转换为1维
	int index = (yIndex*N) + xIndex;

	//中间临时变量
	T temp;

	//划分数据区域
	if (xIndex < N / 2)
	{
		if (yIndex < N / 2)
		{
			//第1块区域（第1区域与第3区域对换数据）
			temp = data[index];

			//First Area
			data[index] = data[index + sEq1];//第3区域的值赋值给第1区域

			//Third Area
			data[index + sEq1] = temp;
		}
	}
	else
	{
		if (yIndex < N / 2)
		{
			//第2区域（第2区域与第4区域对换数据）
			temp = data[index];

			//Second Area
			data[index] = data[index + sEq2];

			//Fourth Area
			data[index + sEq2] = temp;
		}
	}
}
