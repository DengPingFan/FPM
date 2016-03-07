#ifndef CUFFTSHIFTSHARED_H
#define CUFFTSHIFTSHARED_H	

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct kernelConf
{
	dim3 grid;
	dim3 thread;
};

#endif  //CUFFTSHIFTSHARED_H