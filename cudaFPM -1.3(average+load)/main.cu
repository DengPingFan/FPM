#include "stdafx.h"
#include "CmTimer.h"
#include "FPMFunction.h"
#include "cufft.h"

#include "cufftShift_2D.h"
#include "cufftShiftInterface.h"
#include "cufftShiftShared.h"
#include "cufftShift_2D_IP_impl.cu"

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#pragma comment(lib,"cufft.lib")

const int rawImageSize = 128 * 128;

//初始化分配所有Cuda内存
struct Pointers
{
	vecMat lowf;
	vecMat rawimage;

};

//cuda内存指针
__device__ Pointers *pts;


//保存路径参数
string rootDir = "../Data/";//root directory
string resultDir = rootDir + "Result/";//the result directory
string picDir = rootDir + "Pic/";
string rawDir = picDir + "rawImage/";

//显微镜参数
Params param;

//LED采样位置图
Mat kxL(7, 7, CV_8UC1, Scalar(0));
Mat kyL(7, 7, CV_8UC1, Scalar(0));

//模板
Mat mask, not_mask;

//原始采样图像
Mat rawImage(49, rawImageSize, CV_64FC1, Scalar(0));
Mat img(128, 128, CV_64FC1, Scalar(0));//中间图像


//======================================cuda函数声明===============================
cudaError_t countOverLapNumber(int width, int height, CMat centerMask, CMat kxL, CMat kyL, Mat &countMaps);

cudaError_t replaceIntenMeasurement(Mat &lowf, Mat rawImage, Mat mask);

Mat cudaFFT2_IFFT2(Mat lowf, int type);

//cufft逆傅里叶变换后的归一化
void cudaNormalize(double2 *src, double2 *dst, int width, int height);

void cudaConnetPhase(double2 *src, double *src1, double2 *dst, int width, int height);

void cufftCopyByMask(double2 *src, uchar *mask, int width, int height);

//======================================cpu函数声明================================

void Initialize();
void InitializeParams(Mat1d &kx1, Mat1d &ky1);

//Save Partion
void SaveMask(Mat1d kx1, Mat1d ky1, Mat &mask, Mat &not_mask);
void SaveCenterBoundMask(const Mat mask, Mat &centerMask, Mat &boundMask);

void SaveRawImage(CMat kxL, CMat kyL, CMat mask, Mat &rawimage);
void CreateRawimage(CMat kxL, CMat kyL, CMat mask, vecMat &rawimage);
void SaveCountMap(CMat centerMask, Mat &countMaps);

//Load Partion
void LoadParams(Mat &mask, Mat &not_mask, Mat &kxL, Mat &kyL, Mat &countMaps, Mat &boundMask);

void LoadMask(Mat &mask, Mat &not_mask);
void LoadRawImageCountMapBoundMask(Mat &countMaps, Mat &boundMask);
void LoadkxL_kyL(Mat &kxL, Mat &kyL);

void generateLowResolution(CMat f0, CMat mask, Mat &lowf);


//======================================krenel函数实现==================================

//计算某个像素点被多少个圆形区域包含
__global__ void countMap_kernel(const int *kxL, const int*kyL, const double * centerMask, double *countMap, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int i, j;//圆心坐标
	double distance;//当前像素与圆心的距离

	if (x < width && y < height)
	{
		int offset = y * height + x;
		//计算某个像素点被多少个圆包含
		for (int iy = 0; iy <= 6; iy++)
		{
			for (int ix = 6; ix >= 0; ix--)
			{
				i = kxL[iy * 7 + ix] + 64;//位置坐标从左上角加64像素移动到中心
				j = kyL[iy * 7 + ix] + 64;

				distance = (x - i)*(x - i) + (y - j)*(y - j);//计算每个像素分别距离49个圆心位置的距离
				if (centerMask[offset] == 255 && distance < 949)//radius^2=31*31
					countMap[offset]++;
			}
		}
	}
}

//求出src的相位，与src1的振幅部分拼接 
__global__ void connetPhase_kernel(const double2* src, const double *src1, double2 *dst, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height)
	{
		int offset = y*height + x;

		//1.求src相位角
		dst[offset].x = 0.f;//图像实部设为0
		dst[offset].y = atan2(src[offset].y, src[offset].x);//phase angle 相位角atan2(imag/real)

		//2.将src相位角转换为指数形式e^i*phase   
		//eg:EXP(Z) = EXP(X)*(COS(Y)+i*SIN(Y))
		dst[offset].x = cos(dst[offset].y);
		dst[offset].y = sin(dst[offset].y);

		//3.将src相位的指数形式与src1图像振幅部分相乘
		dst[offset].x *= src1[offset];
		dst[offset].y *= src1[offset];

	}

}

//矩阵同除一个数
__global__ void normalize_kernel(const double2 *src, double2 *dst, int width, int height, int scale)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < width && y < height)
	{
		int offset = y * height + x;
		dst[offset].x = src[offset].x / (float)scale;
		dst[offset].y = src[offset].y / (float)scale;
	}
}

__global__ void copyByMask_kernel(double2 *src, const uchar *mask, int width, int  height)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if (x < width && y < height)
	{
		int offset = y * height + x;
		src[offset].x = (mask[offset] == 255) ? src[offset].x : 0;
		src[offset].y = (mask[offset] == 255) ? src[offset].y : 0;
	}
}

int main()
{
	CmTimer tm("Timer ");
	CmTimer tm1("Timer ");
	tm.Start();


	//1.预计算部分（运行一次即可注释掉）
	//Initialize();

	//2.载入参数(0.005s)
	Mat mask, not_mask, countMaps, boundMask;
	LoadParams(mask, not_mask, kxL, kyL, countMaps, boundMask);

	//3.取中间采样图像采用线性插值放大到256*256(0s)
	memcpy(img.data, rawImage.ptr(24), rawImageSize*sizeof(double));
	resize(img, img, Size(256, 256), INTER_LINEAR);
	//normalize(img, img, 1, 0, NORM_MINMAX);

	//4.构造初始高分辨率图像(0.002s)
	Mat f0;
	FPMFunction::fft2(img, f0);
	FPMFunction::fftshift(f0);

	//5.获取边界非重构区域的频域图像(0s)
	Mat boundMaps;
	f0.copyTo(boundMaps, boundMask);

	//6.预先拷贝出49个待拼接的频率区域(0.007s)
	Mat lowf(49, rawImageSize, CV_64FC2);
	generateLowResolution(f0, mask, lowf);

	//*******************************开始运行Cuda程序***********************************
	tm1.Start();
	//7.图像拼接（将49张原始图像分别与对应重构区域的频率拼接）
	replaceIntenMeasurement(lowf, rawImage, mask);
	tm1.Stop();

	//.重构区域更新

	tm.Stop();

	printf("tm1运行时间：%g s\n", tm1.TimeInSeconds());
	printf("程序总运行时间：%g s\n", tm.TimeInSeconds());

	waitKey(0);
	return 0;
}


//=====================================cuda函数实现====================================
//计算每个像素属于圆形区域的次数
cudaError_t countOverLapNumber(int width, int height, CMat centerMask, CMat kxL, CMat kyL, Mat &countMaps)
{
	//分配GPU内存
	int *kxl, *kyl;
	double *countMap;
	double *centerMap;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//分配数据的大小
	size_t size1 = width *height *sizeof(double);
	size_t size2 = 49 * sizeof(int);

	cudaStatus = cudaMalloc((void **)&kxl, size2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&kyl, size2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&countMap, size1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&centerMap, size1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// 从主机中拷贝数据到GPU内存中.
	cudaStatus = cudaMemcpy(kxl, kxL.data, size2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(kyl, kyL.data, size2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(centerMap, centerMask.data, size1, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threads(16, 16);
	int x = (threads.x - 1 + width) / threads.x;//向上取整，计算有多少x
	int y = (threads.y - 1 + height) / threads.y;
	dim3 grids(x, y);

	countMap_kernel << <grids, threads >> >(kxl, kyl, centerMap, countMap, width, height);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "countMap_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching countMap_kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(countMaps.data, countMap, size1, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(kxl);
	cudaFree(kyl);
	cudaFree(centerMap);
	cudaFree(countMap);

	return cudaStatus;
}

//将采样图像rawimage的振幅部分与频域变换后的相位部分拼接【49张同时拼接】
cudaError_t replaceIntenMeasurement(Mat &lowf, Mat rawImage, Mat mask)
{

	//1.将49张lowf进行ifftshift
	const int width = 128;
	const int height = 128;
	int i;//循环变量

	//分配GPU内存
	double2 *dev_lowf, *dst;
	double *dev_rawImage;
	uchar *dev_mask;

	cudaError_t cudaStatus;

	//// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	//	//goto Error;
	//}

	size_t step = width*height;
	size_t sizeDouble2 = step*sizeof(double2);//每张频域图像的尺寸
	size_t size = step*sizeof(double);//每张rawimage图像的尺寸
	size_t mask_size = step*sizeof(uchar);//分配模板空间大小

	//为49张频率空间图像分配内存
	cudaStatus = cudaMalloc((void **)&dev_lowf, 49 * sizeDouble2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//为49张采样图像分配空间
	cudaStatus = cudaMalloc((void **)&dev_rawImage, 49 * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//为拼接结果分配空间
	cudaStatus = cudaMalloc((void **)&dst, 49 * sizeDouble2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//为mask分配空间
	cudaStatus = cudaMalloc((void **)&dev_mask, mask_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//49张采样图像及频域图像拷贝进GPU内存
	cudaStatus = cudaMemcpy(dev_rawImage, rawImage.data, 49 * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(dev_lowf, lowf.data, 49 * sizeDouble2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	//拷贝mask数据
	cudaStatus = cudaMemcpy(dev_mask, mask.data, mask_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	/*vecMat temp;
	for (i = 0; i < 49; i++)
	{
	Mat t(128, 128, CV_64FC2, Scalar(0));
	cudaMemcpy(t.data, dev_lowf + i*step, sizeDouble2, cudaMemcpyDeviceToHost);
	temp.push_back(t);
	}

	Mat p0 = temp[0];
	Mat p1 = temp[1];*/

	//2.将49张lowf进行ifftshift->ifft2

	//cuda plan
	cufftHandle plan;
	cufftPlan2d(&plan, height, width, CUFFT_Z2Z);//CUFFT_Z2Z:complex to complex

	//计算fftShift的内核参数
	kernelConf* conf = (kernelConf*)malloc(sizeof(kernelConf));
	conf = cufftShift::GenAutoConf_2D(width / 2);

	//gpu上进行逆傅里叶变换及连接
	for (i = 0; i < 49; i++)
	{
		//ifftshift
		cufftShift_2D_config_impl(dev_lowf + i*step, width, height, conf);
		//ifft
		cufftExecZ2Z(plan, dev_lowf + i*step, dev_lowf + i*step, CUFFT_INVERSE);
		//normalize
		cudaNormalize(dev_lowf + i*step, dev_lowf + i*step, width, height);

		//combine
		cudaConnetPhase(dev_lowf + i*step, dev_rawImage + i*step, dst + i*step, width, height);

		//fft2
		cufftExecZ2Z(plan, dst + i*step, dev_lowf + i*step, CUFFT_FORWARD);
		//fftshift
		cufftShift_2D_config_impl(dev_lowf + i*step, width, height, conf);

		//拷贝mask区域
		cufftCopyByMask(dev_lowf + i*step, dev_mask, width, height);
	}


	vecMat temp1;
	for (i = 0; i < 49; i++)
	{
		Mat t(128, 128, CV_64FC2, Scalar(0));
		cudaMemcpy(t.data, dev_lowf + i*step, sizeDouble2, cudaMemcpyDeviceToHost);
		temp1.push_back(t);
	}

	Mat p10 = temp1[0];
	Mat p11 = temp1[1];
	Mat p12 = temp1[48];


	////结果拷贝回主机
	//cudaMemcpy(lowf.data, dev_lowf, 49 * sizeDouble2, cudaMemcpyDeviceToHost);

	//3.将49张lowf的相位部分与49张rawImage的振幅部分拼接

	//4.将拼接的49张结果进行fft2变换

	//5.将拼接的49张结果进行fftshift变换

	//6.拷贝回主机

	//7.返回49张重构好的图像

	cudaFree(dev_lowf);
	cudaFree(dev_rawImage);
	cudaFree(dst);

	return cudaStatus;
}

//cufft逆傅里叶变换后的归一化
void cudaNormalize(double2 *src, double2 *dst, int width, int height)
{
	dim3 threads(16, 16);

	int x = (threads.x - 1 + width) / threads.x;//向上取整，计算有多少x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);
	int scale = width*height;
	normalize_kernel << <grids, threads >> >(src, dst, width, height, scale);

	cudaThreadSynchronize();
}

//求rawImage与频域变换后的相位部分拼接
void cudaConnetPhase(double2 *src, double *src1, double2 *dst, int width, int height)
{
	dim3 threads(16, 16);

	int x = (threads.x - 1 + width) / threads.x;//向上取整，计算有多少x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);

	connetPhase_kernel << <grids, threads >> >(src, src1, dst, width, height);

	cudaThreadSynchronize();
}

//拷贝mask区域的数据
void cufftCopyByMask(double2 *src, uchar *mask, int width, int height)
{
	dim3 threads(16, 16);
	int x = (threads.x - 1 + width) / threads.x;//向上取整，计算有多少x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);

	copyByMask_kernel << <grids, threads >> >(src, mask, width, height);

	cudaThreadSynchronize();

}

//==================================cpu函数实现=======================================

//1.初始化
//包括：a.保存和载入数据的路径 
//      b.模板、频域中心重构区域、重构重叠计数模板的保存
//      c.GPU内存分配
void Initialize()
{
	//初始化参数
	Mat1d kx1, ky1;
	InitializeParams(kx1, ky1);

	//保存原形模板
	SaveMask(kx1, ky1, mask, not_mask);

	//保存中心重构区域和边界非重构区域
	Mat centerMask(256, 256, CV_8UC1, Scalar(0));
	Mat boundMask(256, 256, CV_8UC1, Scalar(0));
	SaveCenterBoundMask(mask, centerMask, boundMask);

	//建立原始采样图像
	SaveRawImage(kxL, kyL, mask, rawImage);

	//保存重构区域重叠次数统计图像
	Mat countMaps(256, 256, CV_64FC1, Scalar(0));
	SaveCountMap(centerMask, countMaps);
}

//2.载入参数
//包括：a.载入圆形模板mask,not_mask
//      b.位置图像kxL,kyL
//      c.载入重构区域每个像素重叠次数 countMaps
//      d.非重构区域腌膜
void LoadParams(Mat &mask, Mat &not_mask, Mat &kxL, Mat &kyL, Mat &countMaps, Mat &boundMask)
{
	LoadMask(mask, not_mask);
	LoadRawImageCountMapBoundMask(countMaps, boundMask);
	LoadkxL_kyL(kxL, kyL);
}

void InitializeParams(Mat1d &kx1, Mat1d &ky1)
{
	vecD kx, ky;
	for (double kxVec = -CV_PI / param.a1; kxVec <= CV_PI / param.a1 - 1; kxVec += param.dkx)
		kx.push_back(-kxVec);

	for (double kyVec = -CV_PI / param.a1; kyVec <= CV_PI / param.a1 - 1; kyVec += param.dky)
		ky.push_back(kyVec);

	FPMFunction::meshgrid(Mat(kx), Mat(ky), kx1, ky1);
	FPMFunction::creatkxL_kyL(kxL, kyL);
}

void SaveMask(Mat1d kx1, Mat1d ky1, Mat &mask, Mat &not_mask)
{
	//构造模板
	FPMFunction::creatMask(kx1, ky1, param.r, mask);
	//模板取反
	threshold(mask, not_mask, 250, 255, THRESH_BINARY_INV);

	string path = picDir + "mask.bmp";
	imwrite(path, mask);
	path = picDir + "not_mask.bmp";
	imwrite(path, not_mask);
}

void SaveCenterBoundMask(CMat mask, Mat &centerMask, Mat &boundMask)
{
	FPMFunction::creatCenterBoundMask(mask, kxL, kyL, centerMask, boundMask);
	FPMFunction::matWrite(picDir + "centerMask.raw", centerMask);
	imwrite(picDir + "boundMask.bmp", boundMask);
}

void SaveRawImage(CMat kxL, CMat kyL, CMat mask, Mat &rawimage)
{
	//构建采样图像
	vecMat rawIm;
	CreateRawimage(kxL, kyL, mask, rawIm);

	//定义一个能够包含49张图片的Mat
	//Mat rawImage(49, rawImageSize, CV_64FC1);
	int iP = 0;
	for (int i = 0; i < 49; i++)
		memcpy(rawimage.ptr(iP++), rawIm[i].data, rawImageSize*sizeof(double));

	//保存采样图像
	FPMFunction::matWrite(rawDir + "rawImage.raw", rawimage);
}

void CreateRawimage(CMat kxL, CMat kyL, CMat mask, vecMat &rawimage)
{
	//读取原始图像
	Mat H = imread(picDir + "cameraman.png", 0);

	//进行快速傅里叶变换、中心化
	Mat hp;
	FPMFunction::fft2(H, hp);
	FPMFunction::fftshift(hp);

	//构造49张采样图像
	FPMFunction::creatRawImage(kxL, kyL, hp, mask, rawimage);
}

void SaveCountMap(CMat centerMask, Mat &countMaps)
{
	countOverLapNumber(256, 256, centerMask, kxL, kyL, countMaps);

	vector<Mat> planes;
	Mat countMap2(256, 256, CV_64FC2, Scalar(0));

	split(countMap2, planes);
	planes[0] = planes[1] = countMaps;
	merge(planes, countMap2);

	countMaps = countMap2;
	FPMFunction::matWrite(picDir + "countMaps.raw", countMap2);
}

void LoadMask(Mat &mask, Mat &not_mask)
{
	mask = imread(picDir + "mask.bmp", 0);
	not_mask = imread(picDir + "not_mask.bmp", 0);
}

void LoadRawImageCountMapBoundMask(Mat &countMaps, Mat &boundMask)
{
	FPMFunction::matRead(rawDir + "rawImage.raw", rawImage);
	FPMFunction::matRead(picDir + "countMaps.raw", countMaps);
	boundMask = imread(picDir + "boundMask.bmp", 0);
}

void LoadkxL_kyL(Mat &kxL, Mat &kyL)
{
	FPMFunction::creatkxL_kyL(kxL, kyL);
}

//在频率域中拷贝一系列区域的低分辨率图像
void generateLowResolution(CMat f0, CMat mask, Mat &lowf)
{
	int kxl, kyl;
	int i = 0;
	Rect rec;

	//Mat lowf(49,rawImageSize,CV_64FC2);

	for (int iy = 0; iy <= 6; iy++)
	{
		for (int ix = 6; ix >= 0; ix--)
		{
			kxl = kxL.at<int>(ix, iy);
			kyl = kyL.at<int>(ix, iy);
			rec = Rect(kxl, kyl, 128, 128);

			//高分辨率图像中mask区域拷贝到lowf
			Mat temp;
			f0(rec).copyTo(temp, mask);

			memcpy(lowf.ptr(i++), temp.data, rawImageSize*sizeof(double2));
			//lowf.push_back(temp);
		}
	}
}