// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)
#pragma warning(disable: 4819)


#include <assert.h>
#include <string>
#include <xstring>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <exception>
#include <cmath>
#include <time.h>
#include <set>
#include <queue>
#include <list>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
#include <atlstr.h>
#include <atltypes.h>
#include <omp.h>
#include <strstream>
#include <complex>
#include <math.h>

using namespace std;

#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif


#include <opencv/cv.h>
#include <opencv2/opencv.hpp> 
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)
#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))


#if CV_MAJOR_VERSION == 3
#pragma comment(lib,  cvLIB("imgcodecs")) //the opencv3.0 need it
#endif

#if CV_MAJOR_VERSION == 2
#pragma comment(lib, cvLIB("contrib"))
#endif

#define _S(str) ((str).c_str())

typedef vector<int> veci;
typedef vector<double> vecD;
typedef vector<float> vecF;
typedef vector<cv::Mat> vecMat;
typedef const string CStr;
typedef const cv::Mat CMat;
typedef vector<string> vecS;



// TODO:  在此处引用程序需要的其他头文件