#include "FPMFunction.h"

//I是单通道图像
int FPMFunction::fft2(Mat I, Mat &complexI)
{
	if (I.empty())
		return -1;

	Mat padded;                          //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };

	complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// crop the spectrum, if it has an odd number of rows or columns
	complexI = complexI(Rect(0, 0, complexI.cols & -2, complexI.rows & -2));

	return 0;
}

int FPMFunction::ifft2(Mat I, Mat &matI)
{
	if (I.empty())
		return -1;

	idft(I, matI, DFT_SCALE);

	return 0;
}

bool FPMFunction::fftshift(Mat &src)
{
	if (src.empty())
		return false;

	// rearrange the quadrants of Fourier image  so that the origin is at the image center  ==fftshift()
	int cx = src.cols / 2;
	int cy = src.rows / 2;

	Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	return true;
}

bool FPMFunction::ifftshift(Mat &src)
{
	return fftshift(src);
}

void FPMFunction::AbsAngle(Mat& cmplx64FC2, Mat& mag64FC1, Mat& ang64FC1)
{
	CV_Assert(cmplx64FC2.type() == CV_64FC2);

	mag64FC1.create(cmplx64FC2.size(), CV_64FC1);//magnitude 幅度sqrt(real^2+imag^2)
	ang64FC1.create(cmplx64FC2.size(), CV_64FC1);//phase angle 相位角atan2(imag/real)

	for (int y = 0; y < cmplx64FC2.rows; y++)	{

		const double* cmpD = cmplx64FC2.ptr<double>(y);
		double* dataA = ang64FC1.ptr<double>(y);
		double* dataM = mag64FC1.ptr<double>(y);

		for (int x = 0; x < cmplx64FC2.cols; x++, cmpD += 2)	{
			dataA[x] = atan2(cmpD[1], cmpD[0]);
			dataM[x] = sqrt(cmpD[0] * cmpD[0] + cmpD[1] * cmpD[1]);
		}
	}
}

// Write matrix to binary file
bool FPMFunction::matWrite(CStr& filename, const Mat& _M){
	Mat M;
	_M.copyTo(M);
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M.empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	int headData[3] = { M.cols, M.rows, M.type() };
	fwrite(headData, sizeof(int), 3, file);
	fwrite(M.data, sizeof(char), M.step * M.rows, file);
	fclose(file);
	return true;
}

// Read matrix from binary file
bool FPMFunction::matRead(const string& filename, Mat& _M){
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf, sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", _S(filename));
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);
	M.copyTo(_M);
	return true;
}

void FPMFunction::meshgrid(const Mat &x, const Mat &y, Mat1d &X, cv::Mat1d &Y)
{
	repeat(x.reshape(1, 1), y.total(), 1, X);
	repeat(y.reshape(1, 1).t(), 1, x.total(), Y);
}

//设置固定的圆心移动距离
bool FPMFunction::creatkxL_kyL(Mat &kxL, Mat &kyL)
{
	veci a, b;
	for (int i = 120; i >= 0; i -= 20)
		a.push_back(i);

	for (int i = 0; i <= 120; i += 20)
		b.push_back(i);

	repeat(Mat(a).reshape(1, 1), 7, 1, kxL);
	repeat(Mat(b).reshape(1, 1).t(), 1, 7, kyL);

	return true;
}

bool FPMFunction::creatMask(Mat1d kx1, Mat1d ky1, double radius, Mat &mask)
{
	//构造掩膜
	mask = Mat(256, 256, CV_8U, Scalar(0));
	double x_tmp, y_tmp;

	double r = radius * radius;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			x_tmp = kx1.at<double>(i, j);
			y_tmp = ky1.at<double>(i, j);

			if ((x_tmp * x_tmp + y_tmp * y_tmp) <= r)
				mask.at<uchar>(i, j) = 255;
		}
	}

	//裁剪四周
	mask = mask(Rect(64, 64, 128, 128));
	return true;
}

void FPMFunction::creatCenterBoundMask(CMat mask, CMat kxL, CMat kyL, Mat &centerMask, Mat &boundMask)
{
	int kxl, kyl;
	Rect rec;
	for (int iy = 0; iy <= 6; iy++)
	{
		for (int ix = 6; ix >= 0; ix--)
		{
			kxl = kxL.at<int>(ix, iy);
			kyl = kyL.at<int>(ix, iy);

			rec = Rect(kxl, kyl, 128, 128);

			mask.copyTo(centerMask(rec), mask);
		}
	}
	//阈值分割得到边界模板
	threshold(centerMask, boundMask, 250, 255, THRESH_BINARY_INV);
	centerMask.convertTo(centerMask, CV_64FC1);

}

bool FPMFunction::creatRawImage(CMat kxL, CMat kyL, CMat hp, CMat mask, vecMat &rawimage)
{
	//49张采样原始图像
	//#pragma omp parallel for
	for (int iy = 0; iy <= 6; iy++)
	{
		for (int ix = 6; ix >= 0; ix--)
		{
			Mat hp1;
			hp(Rect(kxL.at<int>(ix, iy), kyL.at<int>(ix, iy), 128, 128)).copyTo(hp1, mask);

			Mat h1;
			ifftshift(hp1);
			ifft2(hp1, h1);

			Mat mag, angle;// mag(幅度)  angle(相位)
			AbsAngle(h1, mag, angle);

			rawimage.push_back(mag);
		}
	}
	return true;
}

