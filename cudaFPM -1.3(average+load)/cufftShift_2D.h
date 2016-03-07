
#include<cufft.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<stdio.h>

namespace cufftShift
{
	//in-place·½Ê½
	void cufftShift_2D(cufftReal* data, int NX, int NY);
	void cufftShift_2D(cufftDoubleReal* data, int NX, int NY);
	void cufftShift_2D(cufftComplex* data, int NX, int NY);
	void cufftShift_2D(cufftDoubleComplex* data, int NX, int NY);

}