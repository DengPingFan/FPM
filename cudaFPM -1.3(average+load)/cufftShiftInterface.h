#include "cufftShiftShared.h"   //kernelConf���������Ҫ�õ�

template <typename T>
extern
void cufftShift_2D_impl(T* input, int NX, int NY);

template <typename T>
extern
void cufftShift_2D_config_impl(T* input, int NX, int NY, kernelConf *conf);
