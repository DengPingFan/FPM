#include "stdafx.h"

using namespace cv;

struct Params{

	double h;
	double a;
	double a1;
	double NA;//数值孔径
	double b;//LED间距
	double lambda;
	double k0;
	double r;
	double z;
	double dkx, dky;
	int N;
	int n;
	int M;
	int n1;
	int rate;



	Params(){

		rate = 2;
		h = 89.74;
		//a = 5.5 / 1000 / 2;
		a = 2.75 / 1000;//um    //In our prototype,the pixel size for each raw image is 2.75um.(原型中的像素大小)
		a1 = a / rate;

		NA = 0.055;//数值孔径  (an enhancement factor of 10)本来是5.5um=0.0055m,现在变成55um=0.055m
		b = 3;//LED间距
		lambda = 632.0 / 1000000;
		k0 = 2 * CV_PI / lambda;
		r = NA*k0;
		N = 256;
		M = N;
		n = N / rate;
		n1 = 7;
		z = 0 / 1000;

		dkx = 2 * CV_PI / (N*a1);
		dky = dkx;
	}

};

struct FPMFunction
{
	static int fft2(Mat I, Mat &matI);

	static int ifft2(Mat I, Mat &matI);

	static bool fftshift(Mat &src);

	static  bool ifftshift(Mat &src);

	static void AbsAngle(Mat& cmplx32FC2, Mat& mag32FC1, Mat& ang32FC1);

	// Write matrix to binary file
	static bool matWrite(CStr& filename, CMat& M);

	// Read matrix from binary file
	static bool matRead(const string& filename, Mat& M);

	static void meshgrid(const Mat &x, CMat &y, Mat1d &X, cv::Mat1d &Y);

	static bool creatkxL_kyL(Mat &kxL, Mat &kyL);

	static bool creatMask(Mat1d kx1, Mat1d ky1, double radius, Mat &mask);

	static void creatCenterBoundMask(CMat mask, CMat kxL, CMat kyL, Mat &centerMask, Mat &boundMask);

	static bool creatRawImage(CMat kxL, CMat kyL, CMat hp, CMat mask, vecMat &rawimage);

};