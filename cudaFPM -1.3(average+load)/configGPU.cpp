#include "configGPU.h"
#include <iostream>


//根据输入的矩阵大小自动选择内核参数
kernelConf * cufftShift::GenAutoConf_2D(int N)
{
	//内核参数配置
	kernelConf * autoConf = (kernelConf*)malloc(sizeof(kernelConf));

	int threadsPerBlock_X;
	int threadsPerBlock_Y;

	if (2 <= N && N < 4)
	{
		threadsPerBlock_X = 2;
		threadsPerBlock_Y = 2;
	}
	if (4 <= N && N < 8)
	{
		threadsPerBlock_X = 4;
		threadsPerBlock_Y = 4;
	}
	if (8 <= N && N < 16)
	{
		threadsPerBlock_X = 8;
		threadsPerBlock_Y = 8;
	}
	if (N >= 16)
	{
		threadsPerBlock_X = 16;
		threadsPerBlock_Y = 16;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int x = (threadsPerBlock_X - 1 + 2 * N) / threadsPerBlock_X;//向上取整，计算有多少x
	int y = (threadsPerBlock_Y - 1 + N) / threadsPerBlock_Y;

	autoConf->thread = dim3(threadsPerBlock_X, threadsPerBlock_Y);
	autoConf->grid = dim3(x, y);

	printf("Auto Block Conf [%d]x[%d]\n", autoConf->thread.x, autoConf->thread.y);
	printf("Auto Grid Conf [%d]x[%d]\n", autoConf->grid.x, autoConf->grid.y);

	return autoConf;

}