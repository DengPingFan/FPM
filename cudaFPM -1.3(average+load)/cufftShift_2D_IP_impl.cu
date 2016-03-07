#include "configGPU.h"
#include "cufftShiftShared.h"
#include "cufftShift_2D_IP.cu"
#include <iostream>

template <typename T>
extern
void cufftShift_2D_impl(T* data, int NX, int NY)
{
	if (NX == NY)
	{
		const int N = NX;
		kernelConf * conf = cufftShift::GenAutoConf_2D(N / 2);
		cufftShift_2D_kernel << < conf->grid, conf->thread >> >(data, N);
	}
	else
	{
		printf("The library is supporting NxN arrays only \n");
		exit(0);
	}
}

template <typename T>
extern
void cufftShift_2D_config_impl(T* data, int NX, int NY, kernelConf* conf)
{
	if (NX == NY)
	{
		const int N = NX;
		cufftShift_2D_kernel << <conf->grid, conf->thread >> >(data, N);
	}
	else
	{
		printf("The library is supporting NxN arrays only \n");
		exit(0);
	}

}

template void cufftShift_2D_impl<cufftReal>
(cufftReal *data, int NX, int NY);

template void cufftShift_2D_impl <cufftDoubleReal>
(cufftDoubleReal* data, int NX, int NY);

template void cufftShift_2D_impl <cufftComplex>
(cufftComplex* data, int NX, int NY);

template void cufftShift_2D_impl <cufftDoubleComplex>
(cufftDoubleComplex* data, int NX, int NY);

template void cufftShift_2D_config_impl <cufftReal>
(cufftReal* data, int NX, int NY, kernelConf* conf);

template void cufftShift_2D_config_impl <cufftDoubleReal>
(cufftDoubleReal* data, int NX, int NY, kernelConf* conf);

template void cufftShift_2D_config_impl <cufftComplex>
(cufftComplex* data, int NX, int NY, kernelConf* conf);

template void cufftShift_2D_config_impl <cufftDoubleComplex>
(cufftDoubleComplex* data, int NX, int NY, kernelConf* conf);

