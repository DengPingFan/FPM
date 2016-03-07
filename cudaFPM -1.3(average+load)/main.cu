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

//��ʼ����������Cuda�ڴ�
struct Pointers
{
	vecMat lowf;
	vecMat rawimage;

};

//cuda�ڴ�ָ��
__device__ Pointers *pts;


//����·������
string rootDir = "../Data/";//root directory
string resultDir = rootDir + "Result/";//the result directory
string picDir = rootDir + "Pic/";
string rawDir = picDir + "rawImage/";

//��΢������
Params param;

//LED����λ��ͼ
Mat kxL(7, 7, CV_8UC1, Scalar(0));
Mat kyL(7, 7, CV_8UC1, Scalar(0));

//ģ��
Mat mask, not_mask;

//ԭʼ����ͼ��
Mat rawImage(49, rawImageSize, CV_64FC1, Scalar(0));
Mat img(128, 128, CV_64FC1, Scalar(0));//�м�ͼ��


//======================================cuda��������===============================
cudaError_t countOverLapNumber(int width, int height, CMat centerMask, CMat kxL, CMat kyL, Mat &countMaps);

cudaError_t replaceIntenMeasurement(Mat &lowf, Mat rawImage, Mat mask);

Mat cudaFFT2_IFFT2(Mat lowf, int type);

//cufft�渵��Ҷ�任��Ĺ�һ��
void cudaNormalize(double2 *src, double2 *dst, int width, int height);

void cudaConnetPhase(double2 *src, double *src1, double2 *dst, int width, int height);

void cufftCopyByMask(double2 *src, uchar *mask, int width, int height);

//======================================cpu��������================================

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


//======================================krenel����ʵ��==================================

//����ĳ�����ص㱻���ٸ�Բ���������
__global__ void countMap_kernel(const int *kxL, const int*kyL, const double * centerMask, double *countMap, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int i, j;//Բ������
	double distance;//��ǰ������Բ�ĵľ���

	if (x < width && y < height)
	{
		int offset = y * height + x;
		//����ĳ�����ص㱻���ٸ�Բ����
		for (int iy = 0; iy <= 6; iy++)
		{
			for (int ix = 6; ix >= 0; ix--)
			{
				i = kxL[iy * 7 + ix] + 64;//λ����������ϽǼ�64�����ƶ�������
				j = kyL[iy * 7 + ix] + 64;

				distance = (x - i)*(x - i) + (y - j)*(y - j);//����ÿ�����طֱ����49��Բ��λ�õľ���
				if (centerMask[offset] == 255 && distance < 949)//radius^2=31*31
					countMap[offset]++;
			}
		}
	}
}

//���src����λ����src1���������ƴ�� 
__global__ void connetPhase_kernel(const double2* src, const double *src1, double2 *dst, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height)
	{
		int offset = y*height + x;

		//1.��src��λ��
		dst[offset].x = 0.f;//ͼ��ʵ����Ϊ0
		dst[offset].y = atan2(src[offset].y, src[offset].x);//phase angle ��λ��atan2(imag/real)

		//2.��src��λ��ת��Ϊָ����ʽe^i*phase   
		//eg:EXP(Z) = EXP(X)*(COS(Y)+i*SIN(Y))
		dst[offset].x = cos(dst[offset].y);
		dst[offset].y = sin(dst[offset].y);

		//3.��src��λ��ָ����ʽ��src1ͼ������������
		dst[offset].x *= src1[offset];
		dst[offset].y *= src1[offset];

	}

}

//����ͬ��һ����
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


	//1.Ԥ���㲿�֣�����һ�μ���ע�͵���
	//Initialize();

	//2.�������(0.005s)
	Mat mask, not_mask, countMaps, boundMask;
	LoadParams(mask, not_mask, kxL, kyL, countMaps, boundMask);

	//3.ȡ�м����ͼ��������Բ�ֵ�Ŵ�256*256(0s)
	memcpy(img.data, rawImage.ptr(24), rawImageSize*sizeof(double));
	resize(img, img, Size(256, 256), INTER_LINEAR);
	//normalize(img, img, 1, 0, NORM_MINMAX);

	//4.�����ʼ�߷ֱ���ͼ��(0.002s)
	Mat f0;
	FPMFunction::fft2(img, f0);
	FPMFunction::fftshift(f0);

	//5.��ȡ�߽���ع������Ƶ��ͼ��(0s)
	Mat boundMaps;
	f0.copyTo(boundMaps, boundMask);

	//6.Ԥ�ȿ�����49����ƴ�ӵ�Ƶ������(0.007s)
	Mat lowf(49, rawImageSize, CV_64FC2);
	generateLowResolution(f0, mask, lowf);

	//*******************************��ʼ����Cuda����***********************************
	tm1.Start();
	//7.ͼ��ƴ�ӣ���49��ԭʼͼ��ֱ����Ӧ�ع������Ƶ��ƴ�ӣ�
	replaceIntenMeasurement(lowf, rawImage, mask);
	tm1.Stop();

	//.�ع��������

	tm.Stop();

	printf("tm1����ʱ�䣺%g s\n", tm1.TimeInSeconds());
	printf("����������ʱ�䣺%g s\n", tm.TimeInSeconds());

	waitKey(0);
	return 0;
}


//=====================================cuda����ʵ��====================================
//����ÿ����������Բ������Ĵ���
cudaError_t countOverLapNumber(int width, int height, CMat centerMask, CMat kxL, CMat kyL, Mat &countMaps)
{
	//����GPU�ڴ�
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

	//�������ݵĴ�С
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

	// �������п������ݵ�GPU�ڴ���.
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
	int x = (threads.x - 1 + width) / threads.x;//����ȡ���������ж���x
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

//������ͼ��rawimage�����������Ƶ��任�����λ����ƴ�ӡ�49��ͬʱƴ�ӡ�
cudaError_t replaceIntenMeasurement(Mat &lowf, Mat rawImage, Mat mask)
{

	//1.��49��lowf����ifftshift
	const int width = 128;
	const int height = 128;
	int i;//ѭ������

	//����GPU�ڴ�
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
	size_t sizeDouble2 = step*sizeof(double2);//ÿ��Ƶ��ͼ��ĳߴ�
	size_t size = step*sizeof(double);//ÿ��rawimageͼ��ĳߴ�
	size_t mask_size = step*sizeof(uchar);//����ģ��ռ��С

	//Ϊ49��Ƶ�ʿռ�ͼ������ڴ�
	cudaStatus = cudaMalloc((void **)&dev_lowf, 49 * sizeDouble2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//Ϊ49�Ų���ͼ�����ռ�
	cudaStatus = cudaMalloc((void **)&dev_rawImage, 49 * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//Ϊƴ�ӽ������ռ�
	cudaStatus = cudaMalloc((void **)&dst, 49 * sizeDouble2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//Ϊmask����ռ�
	cudaStatus = cudaMalloc((void **)&dev_mask, mask_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	//49�Ų���ͼ��Ƶ��ͼ�񿽱���GPU�ڴ�
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

	//����mask����
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

	//2.��49��lowf����ifftshift->ifft2

	//cuda plan
	cufftHandle plan;
	cufftPlan2d(&plan, height, width, CUFFT_Z2Z);//CUFFT_Z2Z:complex to complex

	//����fftShift���ں˲���
	kernelConf* conf = (kernelConf*)malloc(sizeof(kernelConf));
	conf = cufftShift::GenAutoConf_2D(width / 2);

	//gpu�Ͻ����渵��Ҷ�任������
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

		//����mask����
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


	////�������������
	//cudaMemcpy(lowf.data, dev_lowf, 49 * sizeDouble2, cudaMemcpyDeviceToHost);

	//3.��49��lowf����λ������49��rawImage���������ƴ��

	//4.��ƴ�ӵ�49�Ž������fft2�任

	//5.��ƴ�ӵ�49�Ž������fftshift�任

	//6.����������

	//7.����49���ع��õ�ͼ��

	cudaFree(dev_lowf);
	cudaFree(dev_rawImage);
	cudaFree(dst);

	return cudaStatus;
}

//cufft�渵��Ҷ�任��Ĺ�һ��
void cudaNormalize(double2 *src, double2 *dst, int width, int height)
{
	dim3 threads(16, 16);

	int x = (threads.x - 1 + width) / threads.x;//����ȡ���������ж���x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);
	int scale = width*height;
	normalize_kernel << <grids, threads >> >(src, dst, width, height, scale);

	cudaThreadSynchronize();
}

//��rawImage��Ƶ��任�����λ����ƴ��
void cudaConnetPhase(double2 *src, double *src1, double2 *dst, int width, int height)
{
	dim3 threads(16, 16);

	int x = (threads.x - 1 + width) / threads.x;//����ȡ���������ж���x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);

	connetPhase_kernel << <grids, threads >> >(src, src1, dst, width, height);

	cudaThreadSynchronize();
}

//����mask���������
void cufftCopyByMask(double2 *src, uchar *mask, int width, int height)
{
	dim3 threads(16, 16);
	int x = (threads.x - 1 + width) / threads.x;//����ȡ���������ж���x
	int y = (threads.y - 1 + height) / threads.y;

	dim3 grids(x, y);

	copyByMask_kernel << <grids, threads >> >(src, mask, width, height);

	cudaThreadSynchronize();

}

//==================================cpu����ʵ��=======================================

//1.��ʼ��
//������a.������������ݵ�·�� 
//      b.ģ�塢Ƶ�������ع������ع��ص�����ģ��ı���
//      c.GPU�ڴ����
void Initialize()
{
	//��ʼ������
	Mat1d kx1, ky1;
	InitializeParams(kx1, ky1);

	//����ԭ��ģ��
	SaveMask(kx1, ky1, mask, not_mask);

	//���������ع�����ͱ߽���ع�����
	Mat centerMask(256, 256, CV_8UC1, Scalar(0));
	Mat boundMask(256, 256, CV_8UC1, Scalar(0));
	SaveCenterBoundMask(mask, centerMask, boundMask);

	//����ԭʼ����ͼ��
	SaveRawImage(kxL, kyL, mask, rawImage);

	//�����ع������ص�����ͳ��ͼ��
	Mat countMaps(256, 256, CV_64FC1, Scalar(0));
	SaveCountMap(centerMask, countMaps);
}

//2.�������
//������a.����Բ��ģ��mask,not_mask
//      b.λ��ͼ��kxL,kyL
//      c.�����ع�����ÿ�������ص����� countMaps
//      d.���ع�������Ĥ
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
	//����ģ��
	FPMFunction::creatMask(kx1, ky1, param.r, mask);
	//ģ��ȡ��
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
	//��������ͼ��
	vecMat rawIm;
	CreateRawimage(kxL, kyL, mask, rawIm);

	//����һ���ܹ�����49��ͼƬ��Mat
	//Mat rawImage(49, rawImageSize, CV_64FC1);
	int iP = 0;
	for (int i = 0; i < 49; i++)
		memcpy(rawimage.ptr(iP++), rawIm[i].data, rawImageSize*sizeof(double));

	//�������ͼ��
	FPMFunction::matWrite(rawDir + "rawImage.raw", rawimage);
}

void CreateRawimage(CMat kxL, CMat kyL, CMat mask, vecMat &rawimage)
{
	//��ȡԭʼͼ��
	Mat H = imread(picDir + "cameraman.png", 0);

	//���п��ٸ���Ҷ�任�����Ļ�
	Mat hp;
	FPMFunction::fft2(H, hp);
	FPMFunction::fftshift(hp);

	//����49�Ų���ͼ��
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

//��Ƶ�����п���һϵ������ĵͷֱ���ͼ��
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

			//�߷ֱ���ͼ����mask���򿽱���lowf
			Mat temp;
			f0(rec).copyTo(temp, mask);

			memcpy(lowf.ptr(i++), temp.data, rawImageSize*sizeof(double2));
			//lowf.push_back(temp);
		}
	}
}