#include <cufft.h>
#include <iostream>

//Cuda��fftShift�任
template <typename T>
__global__ void cufftShift_2D_kernel(T *data, int N)
{
	int sLine = N;
	int sSlice = N*N;

	//ת������
	int sEq1 = (sSlice + sLine) / 2;
	int sEq2 = (sSlice - sLine) / 2;

	//�߳����꣨2D��
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	//�߳�����ת��Ϊ1ά
	int index = (yIndex*N) + xIndex;

	//�м���ʱ����
	T temp;

	//������������
	if (xIndex < N / 2)
	{
		if (yIndex < N / 2)
		{
			//��1�����򣨵�1�������3����Ի����ݣ�
			temp = data[index];

			//First Area
			data[index] = data[index + sEq1];//��3�����ֵ��ֵ����1����

			//Third Area
			data[index + sEq1] = temp;
		}
	}
	else
	{
		if (yIndex < N / 2)
		{
			//��2���򣨵�2�������4����Ի����ݣ�
			temp = data[index];

			//Second Area
			data[index] = data[index + sEq2];

			//Fourth Area
			data[index + sEq2] = temp;
		}
	}
}
