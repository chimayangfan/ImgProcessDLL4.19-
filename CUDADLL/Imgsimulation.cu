#include"Imgsimulation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <fstream> 
#include <string>
#include <io.h>
#include <vector>
#include <time.h>
#include <thread>
#include <mutex>
#include <atomic>

#include <stdio.h>  
#include <algorithm>

#include "Thread.h"
#include "ThreadPoolExecutor.h"
#include "cuda_profiler_api.h"
#include <helper_cuda.h>//错误处理
#include <helper_string.h>
#include <npp.h>
#include "Endianess.h"
#include "useful.h"
#include "kernel.h"
#include <Windows.h>
#include <GdiPlus.h>
#pragma comment( lib, "GdiPlus.lib" )
using namespace Gdiplus;
using namespace std;
using namespace cv;

double Timedatarefresh = 0.5;
bool   SimulationSuccessFlaf = false;
//线程索申请；
std::mutex gExtrackPointLock;//R类和Rec类的线程安全锁
std::mutex gComressReadDataLock;
std::mutex compress_process_lock;//线程锁申
std::mutex compress_write_lock;//线程锁申
std::mutex compress_writeCPU_lock;//线程锁申

//根据设备性能定义
#define ExtractPointThreads 2
#define CompressionThreads 2
#define CUDAStreams 5
#define GRAYCompressStreams 5
//磁盘剩余存储空间阈值（GB）
#define DiskRemainingSpaceThreshold 50
//根据图片大小定义block和thread个数 
int gHostImgblock = ExtractPointThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
dim3 blocks;												//压缩程序需要的cuda 分块配置
dim3 threads(8, 8);											//压缩程序需要的block 线程数配置
//界面传参结构体
Parameter gStructVarible{ NULL,NULL,NULL,8,1,5120,5120,5120,60,30,300,8,640,640,0,99999,2000,5,0,0 ,4 };
//标志点信息结构体
Infomation SignPoint;
//硬件配置结构体
HardwareInfo HardwareParam;//硬件配置结构体

#define Pretreatment
#ifdef Pretreatment
#define ReadImageNumber 250
#endif // Pretreatment
unsigned char* gHostImage[250] = { NULL };
unsigned char* gHostColorImage[250] = { NULL };

//-------------------------方位盒Model数据-----------------------------//
typedef struct
{
	short RecXmin;
	short RecYmin;
	short RecXmax;
	short RecYmax;
}RecData;//方位盒数据结构
vector<RecData> gHostRecData;//CPU方位盒数据容器
int gRecNum;//方位盒数量（这个是拼图后和规整后的方位盒数量）
int gSingleImgRecNum;//单张图方位盒数量

/*-------------------------数据缓冲数据定义-----------------------*/
struct CircleInfo//特征存储结构体(24字节)
{
	short index;
	short length;
	short area;
	double xpos;
	double ypos;
};

//实时刷新图像
unsigned char * OnlineRefreshIMG;
//通信变量
int  BufferBlockIndex[6] = { 0 };//缓冲区刷新的次数（更新了多少次600张图片）
int  Bufferlength;//每个缓冲区的长度(需要初始化)
vector<int>gWorkingGpuId;//用来存能用设备的设备号
bool ExtractPointInitialSuccessFlag[3] = { false };//用于标记各个提点类是否初始化完成
bool ExtractPointSuccess = false;//实验结束标志位

//矩形盒更新标志位
unsigned char * gRecupImgData = NULL;//这个矩形盒数据更新时所对应的缓冲区（一个图片大小）
bool DevUpdateRec[3] = { false };//当该标志位为true时，表示 CPU端矩形盒数据已经更新完成，GPU端需要拷贝CPU端矩形盒数据至GPU端来更新包围盒子
bool HostUpdateRec = false; //当该标志为true时表示主机端矩形盒子数据更新了
bool RecupdataInitialSuccessFlag = false;

//相机缓冲区
unsigned char * gCameraDress=NULL;
unsigned char * gCameraBuffer[6] = { NULL };
bool CameraBufferFull[6] = { false };//用于通信提点线程，相机内存数据准备就绪

//页锁内存缓冲区(这个用于提点)
unsigned char * gHostBuffer[4] = { NULL };
bool PageLockBufferEmpty[4] = { true };
bool PageLockBufferWorking[4] = { false };
int PageLockBufferStartIndex[4];

//压缩缓冲区（用于压缩）
unsigned char *gHostComressiongBuffer[4] = { NULL };
bool gComressionBufferEmpty[4] = { true };
bool gComressionBufferWorking[4] = { false };
int  gComressionBufferStartIndex[4];


//--------------------------------------------------------开始---------------------------------------------//
/***********************************************************************************************
Function:       RGBtoYUV(核函数）
Description:    只在彩图压缩中使用
将.bmp图像的R、G、B数据转化为压缩所需要的亮度和色差数据，这些数据都在显存中

Calls:          无
Input:          unsigned char* dataIn（显存原图像地址）
int imgHeight（图像高度）
int imgWidth（图像宽度）
unsigned int nPitch（经过对齐后的数据宽度）

Output:        unsigned char* Y, unsigned char* Cb, unsigned char* Cr（得到的亮度和色差数据）
************************************************************************************************/
__global__ void RGBtoYUV(unsigned char* dataIn, unsigned char* Y, unsigned char* Cb, unsigned char* Cr, int imgHeight, int imgWidth, int nPitch, int old_Height, int old_Width)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	if (xIndex < old_Width && yIndex < old_Height) {
		unsigned char blue = dataIn[yIndex *  old_Width * 3 + xIndex * 3 + 2];
		unsigned char green = dataIn[yIndex *  old_Width * 3 + xIndex * 3 + 1];
		unsigned char red = dataIn[yIndex *  old_Width * 3 + xIndex * 3];

		unsigned char y = 0.299 * red + 0.587 * green + 0.114 * blue;
		unsigned char cb = -0.1687 * red - 0.3313 * green + 0.5 * blue + 127;
		unsigned char cr = 0.5 * red - 0.4187 * green - 0.0813 * blue + 127;

		Y[yIndex * nPitch + xIndex] = y;
		Cb[yIndex * nPitch + xIndex] = cb;
		Cr[yIndex * nPitch + xIndex] = cr;
	}

}

/*--------------------------------压缩程序需要的结构体needmemory-------------------------------*/
struct needmemory
{
	Npp16s *pDCT[3] = { 0,0,0 };							//GPU内DCT变换后数据储存变量
	Npp32s DCTStep[3];										//记录字节对齐的DCT变换数据大小
	NppiDCTState *pDCTState;
	Npp8u *pDImage[3] = { 0,0,0 };							//GPU内图像的YCbCr数据
	Npp32s DImageStep[3];									//记录字节对齐的YCbCr数据大小
	Npp8u *pDScan;											//GPU内霍夫曼编码扫描数据
	Npp32s nScanSize;										//pDScan的初始化大小
	Npp8u *pDJpegEncoderTemp;								//GPU内霍夫曼编码的中间数据
	size_t nTempSize;										// pDJpegEncoderTemp的大小
	Npp32s nScanLength;										//霍夫曼编码后的pDScan大小

	Npp8u *hpCodesDC[3];									//标准霍夫曼表的DC、AC值和编码
	Npp8u *hpCodesAC[3];
	Npp8u *hpTableDC[3];
	Npp8u *hpTableAC[3];
};

/*--------------------------------压缩程序需要的结构体needdata-------------------------------*/
struct  needdata
{
	NppiSize oDstImageSize;									//输出的jpg图片大小（长宽值）
	NppiSize aDstSize[3];									//实际压缩图片区域范围
	Npp8u *pdQuantizationTables;							//GPU中的标准量化表
	NppiEncodeHuffmanSpec *apDHuffmanDCTable[3];			// GPU中的霍夫曼直流表
	NppiEncodeHuffmanSpec *apDHuffmanACTable[3];			// GPU中的霍夫曼交流表
};

struct Pk
{
	int Offest;//每个文件的偏移量
	int FileLen;//文件长度
				//int FileNameLen;//文件名长度 
				//char* FileName;//需要打包的文件名
	int FileNumber;
};

class Package
{
public:
	//Package(const char* Fname, int FileNum) :Fname(Fname), FileNum(FileNum)
	Package(const char* Fname) :Fname(Fname)
	{
		table_scale = 0;
		concordancesize = 0;
		head_cache = new char[20000];
		head_bias = 0;
	}

	~Package()
	{
		delete[] head_cache;
		delete[]concordance;
	}
	//void Form_one_head(int index, char* Filename, int FileLen);
	void Package_init(int Num) { FileNum = Num;  concordance = new Pk[FileNum]; }
	void Form_one_head(int index, int one_picture_index, int FileLen);
	void UnPack(const char* name, const char* save_path);											//解包
																									//void Form_total_head();
	void Form_total_head(int one_picture_width, int one_picture_height, int picture_number, int picture_index);



	Pk* concordance;
	fstream file;
	int concordancesize;													//索引表大小
	int FileNum;															//文件个数
	const char* Fname;														//打包完成后的文件名

	char* head_cache;														//头文件总大小
	int head_bias;															//头文件总偏移
	int table_scale;

};

//void Package::Form_one_head(int index, char* Filename, int FileLen)
void Package::Form_one_head(int index, int one_picture_index, int FileLen)
{
	if (index == 0)															//得到每个文件的偏移位置
	{
		concordance[index].Offest = 0;
	}
	else
	{
		concordance[index].Offest = concordance[index - 1].Offest + concordance[index - 1].FileLen;
	}
	//table_scale = table_scale + strlen(Filename) + 1 + 3 * sizeof(int);					//算索引表大小
	table_scale = table_scale + 3 * sizeof(int);
	//concordance[index].FileNameLen = strlen(Filename) + 1;								//文件名大小

	//concordance[index].FileName = new char[50];
	//strcpy(concordance[index].FileName, Filename);
	//cout << concordance[index].FileName << endl;
	concordance[index].FileNumber = one_picture_index;
	concordance[index].FileLen = FileLen;
}

void Package::Form_total_head(int one_picture_width, int one_picture_height, int picture_number, int picture_index)
{
	concordancesize = table_scale + 6 * sizeof(int);					//得到索引表大小
	memcpy(head_cache, (char*)&concordancesize, sizeof(int));
	head_bias += sizeof(int);
	memcpy(head_cache + head_bias, (char*)&FileNum, sizeof(int));
	head_bias += sizeof(int);

	memcpy(head_cache + head_bias, (char*)&one_picture_width, sizeof(int));
	head_bias += sizeof(int);
	memcpy(head_cache + head_bias, (char*)&one_picture_height, sizeof(int));
	head_bias += sizeof(int);
	memcpy(head_cache + head_bias, (char*)&picture_number, sizeof(int));
	head_bias += sizeof(int);
	memcpy(head_cache + head_bias, (char*)&picture_index, sizeof(int));
	head_bias += sizeof(int);
	//cout << FileNum << endl;
	for (int i = 0; i < FileNum; ++i)
	{
		memcpy(head_cache + head_bias, (char*)&concordance[i].Offest, sizeof(int));
		head_bias += sizeof(int);

		memcpy(head_cache + head_bias, (char*)&concordance[i].FileLen, sizeof(int));
		head_bias += sizeof(int);

		//memcpy(head_cache + head_bias, &concordance[i].FileNameLen, sizeof(int));
		memcpy(head_cache + head_bias, &concordance[i].FileNumber, sizeof(int));
		head_bias += sizeof(int);

		//memcpy(head_cache + head_bias, concordance[i].FileName, concordance[i].FileNameLen);
		//head_bias += concordance[i].FileNameLen;
	}
}

void Package::UnPack(const char *name, const char* save_path)										//解包
{
	int one_picture_width, one_picture_height, picture_number, picture_index;
	file.open(name, ios::in | ios::binary);
	file.read((char*)&concordancesize, sizeof(int));					//读取索引表大小
	file.read((char*)&FileNum, sizeof(int));							//读取文件个数

	file.read((char*)&one_picture_width, sizeof(int));
	file.read((char*)&one_picture_height, sizeof(int));
	file.read((char*)&picture_number, sizeof(int));
	file.read((char*)& picture_index, sizeof(int));

	file.seekg(8 + 4 * 4, ios::beg);
	concordance = new Pk[FileNum];
	for (int i = 0; i < FileNum; ++i)									//读取索引表具体的内容
	{
		file.read((char*)&concordance[i].Offest, sizeof(int));			//读取偏移量
		file.read((char*)&concordance[i].FileLen, sizeof(int));			//读取文件大小
		file.read((char*)&concordance[i].FileNumber, sizeof(int));
		//file.read((char*)&concordance[i].FileNameLen, sizeof(int));		//读取文件名大小


		//concordance[i].FileName = new char[concordance[i].FileNameLen];
		//memset(concordance[i].FileName, 0, sizeof(char)*concordance[i].FileNameLen);//设置为零
		//file.read(concordance[i].FileName, concordance[i].FileNameLen);//读取文件名
	}
	fstream file1;
	for (int i = 0; i < FileNum; ++i)
	{
		char arr[1024] = { 0 };
		//sprintf(arr, "%s", concordance[i].FileName);				//另存在文件夹map中
		sprintf_s(arr, "%s\\%d.jpg", save_path, concordance[i].FileNumber);
		file1.open(arr, ios::out | ios::binary);
		file.seekg(concordancesize + concordance[i].Offest, ios::beg);		//打开文件
		for (int j = 0; j < concordance[i].FileLen; ++j)					//copy文件
		{
			file1.put(file.get());
		}

		file1.close();
		Mat img = imread(arr, IMREAD_UNCHANGED);
		for (int j = 0; j < picture_number; j++)
		{
			char one_image_save_path[50];
			sprintf_s(one_image_save_path, "%s\\%d.jpg", save_path, picture_index + i * picture_number + j);

			cv::Rect rect(0, j * one_picture_height / picture_number, one_picture_width, one_picture_height / picture_number);
			Mat image_cut = Mat(img, rect);
			Mat image_copy = image_cut.clone();
			imwrite(one_image_save_path, image_copy);
		}
		//char one_image_save_path[50];
		//sprintf_s(one_image_save_path, "%s\\%d.bin", save_path, picture_index);
	}
	file.close();
	//for (int i = 0; i < FileNum; ++i)//释放内存
	//{
	//delete[]concordance[i].FileName;
	//}
}

unsigned char* gpHudata;									//灰度图片压缩时使用，用来初始化固定的色差值
unsigned char* gpHvdata;

//-------------------------------------标志点提取核函数----------------------------------------//
/*************************************************
函数名称: ColorMakeBorder //

函数描述: 此函数在图像Width方向上对像素数目进行填充，将Width方向填充为128的整数倍；
.         宽度填充计算公式为：int imgWidth = (width + 127) / 128 * 128； //

输入参数：const unsigned char *colorimg ；colorimg是24位彩色图像数据；
.         Parameter devpar；devpar是包含了图像信息的参数；  //

输出参数：unsigned char *dst；dst是在Width方向上填充像素点后的图像数据，填充的点的像素值取值0； //

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.         该核函数倍调用时，线程配置为： block(128,1,1)、 Grid（ImgMakeborderWidth/128, ImgHeight,1）；
.         GPU中一个线程对应处理一个像素点//
*************************************************/
__global__ void   ColorMakeBorder(const unsigned char * colorimg, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//图像列索引
	const int Id_x = blockIdx.y;//图像行索引
	int b = 0;
	int g = 0;
	int r = 0;
	if (Id_y < devpar.ImgWidth)
	{
		b = colorimg[3 * Id_y + Id_x * devpar.ImgWidth *devpar.ImgChannelNum];
		g = colorimg[3 * Id_y + 1 + Id_x * devpar.ImgWidth * devpar.ImgChannelNum];
		r = colorimg[3 * Id_y + 2 + Id_x * devpar.ImgWidth * devpar.ImgChannelNum];
		dst[Id_y + Id_x * devpar.ImgMakeborderWidth] = unsigned char((r * 30 + g * 59 + b * 11 + 50) / 100);
	}
};

/*************************************************
函数名称: GrayMakeBorder //

函数描述: 此函数在图像Width方向上对像素数目进行填充，将Width方向填充为128的整数倍；
.         宽度填充计算公式为：int imgWidth = (width + 127) / 128 * 128； //

输入参数：const unsigned char *src ；Src是灰度图像数据；
.         Parameter devpar；devpar是包含了图像信息的参数；  //

输出参数：unsigned char *dst；dst是在Width方向上填充像素点后的图像数据，填充的点的像素值取值0； //

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.         该核函数倍调用时，线程配置为： block(128,1,1)、 Grid（ImgMakeborderWidth/128, ImgHeight,1）；
.         GPU中一个线程对应处理一个像素点//
*************************************************/
__global__ void  GrayMakeBorder(const unsigned char *src, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//图像列索引
	const int Id_x = blockIdx.y;//图像行索引
	if (Id_y <  devpar.ImgWidth)
	{
		dst[Id_y + Id_x * devpar.ImgMakeborderWidth] = src[Id_y + Id_x * devpar.ImgWidth];
	}
}

/*************************************************
函数名称: Binarization //

函数描述: 函数根据设定的图像阈值，对图像进行二值化；二值化阈值保存在输入参数 Parameter devpar中；
.		  当像素值大于阈值时，将该点像素值置为255；当像素值小于阈值时，将该点像素值置为0； //

输入参数：unsigned char *psrcgray 是灰度图像数据，实参是填充宽度后的灰度图；
.         Parameter devpar 是包含了图像信息参数；     //

输出参数：unsigned char *pdst2val 是二值化结果的数据，实参对应二值图；
.         unsigned char *pdstcounter 是二值化结果的数据副本， 实参对应轮廓图         //

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.         该核函数倍调用时，线程配置为： block(128,1,1)、 Grid（ImgMakeborderWidth/128, ImgHeight,1）；
.         GPU中一个线程对应处理一个像素点 ；    //

*************************************************/
__global__ void Binarization(unsigned char *psrcgray, unsigned char *pdst2val, unsigned char *pdstcounter, Parameter devpar)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//线程号索引
	int temp = int(psrcgray[Id]);//寄存器保存像素，提高访存效率								
	if (Id < devpar.ImgMakeborderWidth * devpar.ImgHeight*devpar.PictureNum)//边界保护
	{
		pdst2val[Id] = unsigned char(255 * int(temp>devpar.Threshold));//二值化
		pdstcounter[Id] = unsigned char(255 * int(temp>devpar.Threshold));
	}
}

/*************************************************
函数名称: Dilation  //

函数描述: 函数对二值化图进行8邻域膨胀操作，即若某一个点像素值为0的点的八邻域内有非0像素点，则将该点置为255； //

输入参数：unsigned char *psrc 是二值化图数据，该参数作用是作为腐蚀操作的模板副本；
.         Parameter devpar 是包含了图像信息参数；     //

输出参数：unsigned char *pdst 是腐蚀操作结果的数据，实际调用时，该参数输入二值化图数据，通过膨胀操作对其进行更新；   //

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.         该核函数倍调用时，线程配置为： block(128,1,1)、 Grid（ImgMakeborderWidth/128, ImgHeight,1）；
.         GPU中一个线程对应处理一个像素点 ；    //

*************************************************/
__global__  void Dilation(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y代表列索引
	const int Id_x = blockIdx.y;//Id_x代表行信息  
	int temp;//临时变量：用于累加八邻域像素值
	if (Id_y> 1 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x < devpar.PictureNum*devpar.ImgHeight - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] == 0)
		{
			temp = int(psrc[Id_y - 1 + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x - 1)* devpar.ImgMakeborderWidth])
				+ int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y - 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]);
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp > 0 ? 255 : 0;//膨胀操作
		}
	}
}

/*************************************************
函数名称: Erosion  //

函数描述: 函数对膨胀操作后的图进行4邻域腐蚀操作，即若某一个点像素值为255的点的4邻域（十字架邻域）内有0像素点，则将该点置为0； //

输入参数：unsigned char *psrc 是膨胀操作后的图像数据；
.         Parameter devpar 是包含了图像信息参数；     //

输出参数：unsigned char *pdst 是腐蚀操作结果的数据，即标志点轮廓图。
.         实际调用时，该参数输入膨胀操作后的图像数据，通过腐蚀操作对其进行更新；//

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.         该核函数倍调用时，线程配置为： block(128,1,1)、 Grid（ImgMakeborderWidth/128, ImgHeight,1）；
.         GPU中一个线程对应处理一个像素点 ；    //

*************************************************/
__global__  void Erosion(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y代表行信息
	const int Id_x = blockIdx.y;//Id_x代表列信息
	int temp;//临时变量累加4邻域像素值
			 //利用4领域值掏空内部点，提取轮廓信息，现在的dst就是存储轮廓的信息
	if (Id_y > 0 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x <devpar.ImgHeight*devpar.PictureNum - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] != 0)
		{
			temp = int(psrc[Id_y + (Id_x - 1)*devpar.ImgMakeborderWidth]) + int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)*devpar.ImgMakeborderWidth]);//用4领域腐蚀
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp >= 1020 ? 0 : 255;//腐蚀操作
		}
	}
}

/*************************************************
函数名称: GetCounter  //

函数描述: 根据输入轮廓图，利用8邻域追踪法提取标志点的周长和包围盒；  //
.
输入参数：unsigned char *psrc 是轮廓图数据；
.         Parameter devpar 是包含了图像信息参数；     //

输出参数：short *c_length 是提取的标志点周长，当提取失败时，将周长特征置为0；
.         x_min、y_min、x_max、y_max是标志点的包围盒数据，包围盒是一个与标志点相切的矩形，包围盒
.         数据包括矩形的左上角坐标（x_min，y_min）和右下角坐标（x_max，y_max）；
.		  当特征提取失败时，将包围盒数据置0；

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.		  GPU中一个线程对应处理一个图像块，一个线程至多提取出一个标志点的特征信息。
.         该核函数倍调用时，线程配置为： block(128,1,1)、Grid(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1)；
.		  其中ColThreadNum和RowThreadNum分别是对图像进行分块后的列块数和行块数，且对列方向图像块数目ColThreadNum
.		  进行了填充，填充为了128的整数倍；
.         图像块大小一般为PicBlockSize×PicBlockSize，其中PicBlockSize取值一般为8、16、32；

*************************************************/
__global__  void GetCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max, Parameter devpar)
{
	/*八零域方向数组，用于更新轮廓点,初始化方向为正右方（0号位），顺时针旋转45°（索引加1）*/
	const  int direction_y[8] = { 1,1,0,-1,-1,-1,0,1 };
	const  int direction_x[8] = { 0,1,1,1,0,-1,-1,-1 };

	//short Picblocksize = devpar.PicBlockSize;//获取图像块大小
	/*获取行列索引号*/
	const int y = (blockIdx.x*blockDim.x + threadIdx.x) * devpar.PicBlockSize;//y代表列索引
	const int x = blockIdx.y * devpar.PicBlockSize;//x代表行索引
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//线程号
																						 /*初始化输出结果值*/
	c_length[Id] = 0;
	x_min[Id] = 0;
	x_max[Id] = 0;
	y_min[Id] = 0;
	y_max[Id] = 0;
	bool SuccessFlag = false;//用于标记提点是否成功，当为true时，表示当前线程块已经成功提取到一个标志点特征
							 /*初始化包围盒数据*/
	short Rec_xmx = 0, Rec_xmm = 0;
	short Rec_ymx = 0, Rec_ymm = 0;

	if ((y / devpar.PicBlockSize) < (devpar.ImgWidth / devpar.PicBlockSize) && (x / devpar.PicBlockSize) < (devpar.ImgHeight*devpar.PictureNum / devpar.PicBlockSize))//边界判断
	{
		for (int i = x; i < (x + devpar.PicBlockSize); i++)
		{
			for (int j = y; j < (y + devpar.PicBlockSize); j++)
			{
				if (255 == src[j + i * devpar.ImgMakeborderWidth])
				{
					/*初始化包围盒数据*/
					Rec_ymx = j;
					Rec_ymm = j;
					Rec_xmx = i;
					Rec_xmm = i;

					/*定义根节点*/
					short root_x = i;//行索引
					short root_y = j;//列索引
					short counts;//用于8邻域循环计数
					short curr_d = 0;//方向数组索引计数，取值0-7表示八零域的8各不用的方位

									 /*进行跟踪*/
					for (short cLengthCount = 2; cLengthCount < devpar.LengthMax; cLengthCount++)//
					{
						/*定义根标记点*/
						short boot_x = root_x;
						short boot_y = root_y;

						/*更新方位盒数据*/
						Rec_xmx = Rec_xmx > root_x ? Rec_xmx : root_x;
						Rec_ymx = Rec_ymx > root_y ? Rec_ymx : root_y;
						Rec_xmm = Rec_xmm < root_x ? Rec_xmm : root_x;
						Rec_ymm = Rec_ymm < root_y ? Rec_ymm : root_y;

						/*搜索根节点的八邻域点*/
						for (counts = 0; counts < 8; counts++)
						{
							/*防止索引出界*/
							curr_d -= curr_d >= 8 ? 8 : 0;
							curr_d += curr_d < 0 ? 8 : 0;

							/*事实上，只需要判断7个领域内的信息(除了第一次之外)，当count=6时刚好循环到上一个轮廓点*/
							if (cLengthCount >2 && (counts == 6))
							{
								curr_d++;
								continue;
							}

							/*获取邻域点boot*/
							boot_x = root_x + direction_x[curr_d];//更新行索引
							boot_y = root_y + direction_y[curr_d];//更新列索引

							/*判断点是否越界，超过图像的索引区域*/
							if (boot_x < 0 || boot_x >= devpar.ImgHeight*devpar.PictureNum || boot_y < 0 || boot_y >= devpar.ImgWidth)
							{
								curr_d++;
								continue;
							}
							/*如果存在边缘*/
							if (255 == src[boot_y + boot_x * devpar.ImgMakeborderWidth])
							{
								curr_d -= 2;   //更新当前方向  
								root_x = boot_x;//更新根节点
								root_y = boot_y;
								break;
							}
							curr_d++;
						}   // end for  

							/*边界条件判断*/
						if (8 == counts || (root_x >= (x + devpar.PicBlockSize) && root_y >= (y + devpar.PicBlockSize)))
						{
							break;
						}
						/*正常结束*/
						if (root_y == j && root_x == i)
						{
							x_min[Id] = Rec_xmm;
							x_max[Id] = Rec_xmx;
							y_min[Id] = Rec_ymm;
							y_max[Id] = Rec_ymx;
							c_length[Id] = cLengthCount;
							SuccessFlag = true;
							break;
						}//正常结束if
					}//外围for结束			
				}//判断前景点if结束
				if (SuccessFlag)
					break;
				j = Rec_ymx > j ? Rec_ymx : j;//更新列方向搜索步长
			}//第一个for结束
			if (SuccessFlag)
				break;
			i = Rec_xmx > i ? Rec_xmx : i;//更新行方向搜索步长
		}//第二个for 结束
	}
}//核函数结束

 /*筛选方位盒*/
__global__ void SelectTrueBox(unsigned char *ImgCounter, short *clength, short* Recxmm, short * Recymm, short* Recxmx, short *Recymx, short*index, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	index[Id] = 0;
	short temp1 = 0;
	short yMidPos = 0;
	short xMidPos = 0;
	short Rxmm = Recxmm[Id];
	short Rymm = Recymm[Id];
	short RecBoxHeight = Recxmx[Id] - Recxmm[Id];
	short RecBoxWidth = Recymx[Id] - Recymm[Id];

	if (clength[Id] > devpar.LengthMin)
	{
		if ((float(RecBoxHeight) / float(RecBoxWidth))<1.5&& float((RecBoxHeight) / float(RecBoxWidth)) >0.7)//方位盒长款比怎么确定
		{
			if (Rxmm > 0 && Rymm > 0 && Recxmx[Id] < devpar.ImgHeight*devpar.PictureNum - 1 && Recymx[Id] < devpar.ImgWidth - 1)
			{
				yMidPos = Rymm + RecBoxWidth / 2;//中心坐标
				xMidPos = Rxmm + RecBoxHeight / 2;//中心坐标
				for (int i = -1; i < 2; i++)//看矩形盒子中心9领域是否有点
				{
					if (xMidPos + 1 < devpar.ImgHeight*devpar.PictureNum&&yMidPos + 1 < devpar.ImgWidth)
					{
						temp1 += ImgCounter[yMidPos - 1 + (xMidPos + i)*devpar.ImgMakeborderWidth];
						temp1 += ImgCounter[yMidPos + (xMidPos + i)*devpar.ImgMakeborderWidth];
						temp1 += ImgCounter[yMidPos + 1 + (xMidPos + i)*devpar.ImgMakeborderWidth];
					}
				}
				for (int i = 0; Rxmm + i <= Rxmm + RecBoxHeight - i; i++)//判断Height方向
				{
					temp1 += ImgCounter[yMidPos + (Rxmm + i)*devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
					temp1 += ImgCounter[yMidPos + (Rxmm + RecBoxHeight - i)*devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
				}
				for (int i = 0; Rymm + i <= Rymm + RecBoxWidth - i; i++)//判断width方向
				{
					temp1 += ImgCounter[Rymm + i + xMidPos * devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
					temp1 += ImgCounter[Rymm + RecBoxWidth - i + xMidPos * devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
				}
				index[Id] = temp1 > 4 ? 0 : 1;
			}
		}
	}
}

__global__  void SelectNonRepeatBox(short* Recxmm, short * Recymm, short*index, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//获取线程索引号
	short temp = 0;//建立临时变量，用于表示当前块提取的特征是否应当删除
	if (index[Id] != 0)
	{
		if ((Id > devpar.ColThreadNum) && (Id < devpar.ColThreadNum*(devpar.RowThreadNum - 1)))//边界判定
		{
			if (Recxmm[Id] != 0)//判断当前块提取特征是否有效
			{
				/*判断一个图像块获取的坐标是否和与它相邻的右图像块（列+1）、下图像块（行+1）和右上图像块（行-1，列+1）获取的坐标一致*/
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id + 1])) && (Recymm[Id] == Recymm[Id + 1])) ? 1 : 0;//右
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id + devpar.ColThreadNum])) && (short(Recymm[Id]) == short(Recymm[Id + devpar.ColThreadNum]))) ? 1 : 0;//下
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id - devpar.ColThreadNum + 1])) && (short(Recymm[Id]) == short(Recymm[Id - devpar.ColThreadNum + 1]))) ? 1 : 0;//右上
				index[Id] = temp > 0 ? 0 : 1;//输出特征有效标志
			}
		}
	}
}

__global__  void GetNonRepeatBox(short *Recxmm, short *Recymm, short*index, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//线程索引
	const int y = blockIdx.x*blockDim.x + threadIdx.x;//列块数索引
	const int x = blockIdx.y;//行块数索引
	int Id2 = 0;
	if (index[Id] != 0)
	{
		for (int i = x - 4; i < x + 4; i++)
			for (int j = y - 4; j < y + 4; j++)
				if (j > 0 && j < devpar.ImgWidth / devpar.PicBlockSize&&i > 0 && i < devpar.ImgHeight*devpar.PictureNum / devpar.PicBlockSize)
				{
					Id2 = j + i * devpar.ColThreadNum;
					if (index[Id2] != 0)
					{
						if ((short(Recxmm[Id]) == short(Recxmm[Id2])) && (short(Recymm[Id]) == short(Recymm[Id2])))
						{
							index[Id] = Id > Id2 ? 0 : 1;
						}
					}
				}
	}
}

/*************************************************
函数名称: GetInfo  //

函数描述: 根据方位盒信息和输入灰度图像，提取标志点重心和面积特征   //
.
输入参数：unsigned char* src_gray 是灰度图像；
.         short *length 是 提取出的周长特征，当length>LengthMin 时，表示提取出的方位盒信息有效；
.         x_min、y_min、x_max、y_max是标志点的包围盒数据；
.         Parameter devpar 是包含了图像信息参数；     //

输出参数：short *xpos、short*ypos 是利用灰度重心法提取出的标志点重心坐标；
.         short *area  是提取出来的面积特征
.		   当方位盒数据无效时，将short *xpos、short*ypos、short *area都置0；

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.		   GPU中一个线程对应处理一个图像块的方位盒数据；
.         该核函数倍调用时，线程配置与GetCounter函数一致： block(128,1,1)、Grid(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1)；

*************************************************/
__global__  void GetInfo(unsigned char* src_gray, short *index, short* x_min, short * y_min, short* x_max, short *y_max, double *xpos, double*ypos, short *area, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short myArea = 0;
	double sum_gray = 0;//圆点区域的灰度值之和
	double x_sum = 0;//x灰度值加权和
	double y_sum = 0;//y灰度值加权和
	short mThreshold = devpar.Threshold;//二值化阈值
	xpos[Id] = 0;
	ypos[Id] = 0;
	int xRealIndex = 0;
	//保存方位盒边界
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = (ymx - ymm + 3) / 4 * 4;
	unsigned char temp0, temp1, temp2, temp3;//用寄存器暂存图像数据，减小全局内存的访问，提高访存效率

	if (index[Id] >0)
	{
		//循环优化,这种情况会多计算一些区域的值（需要处理一下）
		for (int i = x_min[Id]; i <= x_max[Id]; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
				xRealIndex = i%devpar.ImgHeight;
				//防止越界
				temp0 = j > ymx ? 0 : 1;  //qwt
				temp1 = j + 1 > ymx ? 0 : 1;
				temp2 = j + 2 > ymx ? 0 : 1;
				temp3 = j + 3 > ymx ? 0 : 1;
				//根据二值化阈值 
				temp0 *= src_gray[j   *temp0 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
				temp1 *= src_gray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
				temp2 *= src_gray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
				temp3 *= src_gray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;
				myArea += temp0 > 0 ? 1 : 0;//面积累加
				myArea += temp1 > 0 ? 1 : 0;
				myArea += temp2 > 0 ? 1 : 0;
				myArea += temp3 > 0 ? 1 : 0;
				sum_gray += temp0 + temp1 + temp2 + temp3;
				x_sum += xRealIndex* temp0 + xRealIndex * temp1 + xRealIndex * temp2 + xRealIndex * temp3;
				y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;
			}
		index[Id] = (myArea > devpar.AreaMin&&myArea < devpar.AreaMax) ? 1 : 0;
		area[Id] = myArea;
		xpos[Id] = x_sum / sum_gray;
		ypos[Id] = y_sum / sum_gray;
	}
}

/*************************************************
函数名称: GetRecInfo  //

函数描述: 矩形模式的特征提取函数；根据预提取的包围盒数据、灰度图和轮廓图，提取标志点的特征信息   //

输入参数：RecData* mRec  预提取的方位盒数据
.         unsigned char *psrcgray  灰度图数据
.		  unsigned char *psrccounter 轮廓图数据
.	      Parameter devpar   图像信息结构体                          //

输出参数：short *length    周长特征
.         short* area      面积特征
.         short *xpos, short *ypos    重心坐标

返回值  : 无    //

其他说明: 函数为核函数，在主机端调用，设备端执行；
.		  GPU中一个线程对应处理一个图像块的方位盒数据；
.         核函数的线程配置为block(128,1,1)	Grid(Gridsize, 1, 1);其中Gridsize= mRecCount / 128,mRecCount为预提取的包围盒数量,
.		  在预提取包围盒时，对包围盒数量进行了填充，填充为了128的整数倍
*************************************************/
__global__	void GetRecInfo(RecData* mRec, unsigned char *psrcgray, unsigned char *psrccounter,
	short *length, short* area, double *xpos, double *ypos, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x;//获取线程号
	int mThreshold = devpar.Threshold;//二值化阈值
	short myArea = 0;//用于面积计数
	int clengthCount = 0;//计算周长的临时变量
	short clength = 0;//周长计数
	double sum_gray = 0;//圆点区域的灰度值之和
	double x_sum = 0;//x灰度值加权和
	double y_sum = 0;//y灰度值加权和
	int xRealIndex = 0;
					 /*读取方位盒*/
	short xmm = mRec[Id].RecXmin;
	short xmx = mRec[Id].RecXmax;
	short ymm = mRec[Id].RecYmin;
	short ymx = mRec[Id].RecYmax;
	short jcount = (ymx - ymm + 3) / 4 * 4;//列向循环次数规整
	unsigned char temp0, temp1, temp2, temp3;//temp保存灰度图像数据临时变量（用寄存器储存图像数据，提高访问速度）
	unsigned char t0, t1, t2, t3;//t用于保存轮廓图像数据临时变量

								 /*输出特征初始化*/
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	length[Id] = 0;

	for (int i = xmm; i <= xmx; i++)
		for (int j = ymm; j <= ymm + jcount; j = j + 4)
		{
			xRealIndex = i%devpar.ImgHeight;
			/*防止越界*/
			temp0 = j    > ymx ? 0 : 1;
			temp1 = j + 1> ymx ? 0 : 1;
			temp2 = j + 2> ymx ? 0 : 1;
			temp3 = j + 3> ymx ? 0 : 1;

			t0 = temp0;//qwt
			t1 = temp1;
			t2 = temp2;
			t3 = temp3;

			/*读取列向相邻4个像素点像素值*/
			temp0 *= psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
			temp1 *= psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
			temp2 *= psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
			temp3 *= psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;

			t0 *= psrccounter[j   *t0 + i * devpar.ImgMakeborderWidth];
			t1 *= psrccounter[(j + 1)*t1 + i * devpar.ImgMakeborderWidth];
			t2 *= psrccounter[(j + 2)*t2 + i * devpar.ImgMakeborderWidth];
			t3 *= psrccounter[(j + 3)*t3 + i * devpar.ImgMakeborderWidth];


			myArea += temp0 > 0 ? 1 : 0; //面积计算
			myArea += temp1 > 0 ? 1 : 0;
			myArea += temp2 > 0 ? 1 : 0;
			myArea += temp3 > 0 ? 1 : 0;


			clengthCount += t0 + t1 + t2 + t3;//周长计算

			sum_gray += temp0 + temp1 + temp2 + temp3;//灰度累加
			x_sum += xRealIndex* temp0 + xRealIndex * temp1 + xRealIndex * temp2 + xRealIndex * temp3;
			y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;//y灰度加权累加
		}
	clength = clengthCount / 255;//计算周长
								 /*输出特征*/
	length[Id] = clength;
	area[Id] = myArea;
	xpos[Id] = x_sum / sum_gray;
	ypos[Id] = y_sum / sum_gray;

}

//-------------------------------------------------------结束----------------------------------------//

//-------------------------------------灰度图像压缩核函数----------------------------------------//
/**
* 常量存储器中的值分解（输入范围从-4096到4095…（包括这两种）从系数值映射到值的代码中，以确定其位大小。
*/
__device__ unsigned int GPUjpeg_huffman_value[8 * 1024];
/**
* H
* huffman编码表- 每一种编码表都有257个成员 (256 + 1 extra)
* 依次包括以下四个huffman编码表:
*    - luminance (Y) AC
*    - luminance (Y) DC
*    - chroma (cb/cr) AC
*    - chroma (cb/cr) DC
*/
__device__ uint32_t gpujpeg_huffman_gpu_tab[(256 + 1) * 4];

dim3 gpujpeg_huffman_encoder_grid_size(int tblock_count)
{
	dim3 size(tblock_count);
	while (size.x > 0xffff) {
		size.x = (size.x + 1) >> 1;
		size.y <<= 1;
	}
	return size;
}
/* 内部函数实现 */
static int ALIGN(int x, int y) {  //取y的整数倍
								  // y must be a power of 2.
	return (x + y - 1) & ~(y - 1);
}

/***********************************************************************************************************
/***函数名称：write_bitstream
/***函数功能：将bitstream写入到图像bit流d_JPEGdata中去
/***输    入：bit_location  每个mcu图像单元编码得到的bit流开始的位置
/***输    入：bit_length    每个mcu图像单元编码得到的bit流位长度
/***输    入：bit_code      每个mcu图像单元每个非零数字编码得到的huffman编码
/***输    出：d_JPEGdata    用于存储图像数据编码得到的最终bitstream
/***返    回：无返回
************************************************************************************************************/
__device__ void write_bitstream(unsigned int even_code, unsigned int odd_code, int length, int bit_location, int even_code_size, BYTE *d_JPEGdata) {
	//将一个线程的数据编码写入数据编码缓存空间
	const int byte_restbits = (8 - (bit_location & MASK));
	const int byte_location = bit_location >> SHIFT;
	int write_bytelocation = byte_location;
	uint64_t  threadwrite_code = ((uint64_t)even_code << (24 + byte_restbits)) + ((uint64_t)odd_code << (24 + byte_restbits - even_code_size));
	int right_shift = 56;
	if (byte_restbits != 8) {
		write_bytelocation++;
		length -= byte_restbits;
		right_shift -= 8;
	}
	for (int i = length; i > 0; i = i - 8) {
		d_JPEGdata[write_bytelocation] = (threadwrite_code >> right_shift) & 0XFF;
		right_shift -= 8;
		write_bytelocation++;
	}
	if (byte_restbits != 8) {
		d_JPEGdata[byte_location] = d_JPEGdata[byte_location] | (threadwrite_code >> 56) & 0XFF;
	}
}

/**
*初始化huffman编码的数据，形成常量数据编码表
*/
__global__ static void
GPUjpeg_huffman_encoder_value_init_kernel() {
	// fetch some value
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int value = tid - 4096;

	// decompose it
	unsigned int value_code = value;
	int absolute = value;
	if (value < 0) {
		// valu eis now absolute value of input
		absolute = -absolute;
		// For a negative input, want temp2 = bitwise complement of abs(input)
		// This code assumes we are on a two's complement machine
		value_code--;
	}

	// 计算编码数据的bit位数
	unsigned int value_nbits = 0;
	while (absolute) {
		value_nbits++;
		absolute >>= 1;
	}
	//将数据结果存于表中 (编码数据的值存在高位，左对齐；编码数据的bit位数存在低位右对齐)
	GPUjpeg_huffman_value[tid] = value_nbits | (value_code << (32 - value_nbits));
}

__device__ static unsigned int
gpuhuffman_encode_value(const int preceding_zero_count, const int coefficient,
	const int huffman_lut_offset) {
	// 读取编码数据的huffman编码
	const unsigned int packed_value = GPUjpeg_huffman_value[4096 + coefficient];

	// 将packed_value分解成编码和编码bit位长度
	const int value_nbits = packed_value & 0xf;
	const unsigned int value_code = packed_value & ~0xf;

	// find prefix of the codeword and size of the prefix
	const int huffman_lut_idx = huffman_lut_offset + preceding_zero_count * 16 + value_nbits;
	const unsigned int packed_prefix = gpujpeg_huffman_gpu_tab[huffman_lut_idx];
	const unsigned int prefix_nbits = packed_prefix & 31;

	// 返回编码数据的编码和它的编码长度
	return (packed_prefix + value_nbits) | (value_code >> prefix_nbits);
}

__global__ static void
gpujpeg_huffman_gpu_encoder_encode_block(BSI16 *d_ydst, int MCU_total, BYTE *d_JPEGdata,
	int *prefix_num, int offset, const int huffman_lut_offset) {
	//计算对应的图像block id号	
	const int block_idx = (blockIdx.y * gridDim.x << 2) + (blockIdx.x << 2) + threadIdx.y;
	if (block_idx >= MCU_total) return;

	__shared__ int Length_count[(THREAD_WARP + 1) * 4];
	d_ydst += block_idx << 6;
	const int load_idx = threadIdx.x * 2;
	int in_even = d_ydst[load_idx];
	const int in_odd = d_ydst[load_idx + 1];

	//对直流分量进行差分编码
	if (threadIdx.x == 0 && block_idx != 0) in_even = in_even - d_ydst[load_idx - 64];
	if (threadIdx.x == 0 && block_idx == 0) in_even = in_even - 64;

	//计算当前编码数据前面0的个数
	const unsigned int nonzero_mask = (1 << threadIdx.x) - 1;
	const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC数据都看作是非零数据
	const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
	const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
	const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);

	//计算当前线程偶编码数据编码前面0的个数
	int zeros_before_even = 2 * (zero_pair_count + threadIdx.x - 32);
	if ((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
		zeros_before_even += 1;
	}

	// true if any nonzero pixel follows thread's odd pixel
	const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;

	// 计算奇数位编码数据前面的编码 ,如果交流分量in_even是0，则in_odd前面的0的个数+1
	// (the count is actually multiplied by 16)
	int zeros_before_odd = (in_even || !threadIdx.x) ? 0 : zeros_before_even + 1;

	// clear zero counts if no nonzero pixel follows (so that no 16-zero symbols will be emited)
	// otherwise only trim extra bits from the counts of following zeros
	const int zero_count_mask = nonzero_follows ? 0xF : 0;
	zeros_before_even &= zero_count_mask;
	zeros_before_odd &= zero_count_mask;

	int even_lut_offset = huffman_lut_offset;
	if (0 == threadIdx.x) {
		// first thread uses DC part of the table for its even value
		even_lut_offset += 256 + 1;
	}

	// 一个block的结束标志
	if (0 == ((threadIdx.x ^ 31) | in_odd)) {
		// 如果需要添加结束标志，则将zeros_before_odd的值改为16
		zeros_before_odd = 16;
	}

	// each thread gets codeword for its two pixels
	unsigned int even_code = gpuhuffman_encode_value(zeros_before_even, in_even, even_lut_offset);
	unsigned int odd_code = gpuhuffman_encode_value(zeros_before_odd, in_odd, huffman_lut_offset);

	int *bl_ptr = Length_count + (THREAD_WARP + 1) * threadIdx.y;
	const unsigned int even_code_size = even_code & 31;
	const unsigned int odd_code_size = odd_code & 31;
	int bit_length = even_code_size + odd_code_size;
	even_code = even_code & ~31;
	odd_code = odd_code & ~31;
	int code_nbits = bit_length;

	//计算每个BLOCK中非零编码的数据个数
	unsigned int prefix_bitmap = __ballot(bit_length);
	int prefix_count = __popc(prefix_bitmap & nonzero_mask);
	if (bit_length) {
		bl_ptr[prefix_count] = bit_length;
		__syncthreads();
		//进行前缀求和运算
		for (int j = 0; j < prefix_count; j++) {
			code_nbits = code_nbits + bl_ptr[j];
		}
	}
	if (threadIdx.x == 31) {
		prefix_num[block_idx * 3 + offset] = code_nbits;
	}
	//计算写入缓存区的具体字节位置，确定写入d_JPEGdata的位置
	BYTE *Write_JPEGdata = d_JPEGdata + (block_idx << 6);
	const int bit_location = code_nbits - bit_length;
	const int byte_restbits = (8 - (bit_location & MASK));
	const int byte_location = bit_location >> SHIFT;
	int write_bytelocation = byte_location;
	//将一个线程的数据编码写入数据编码缓存空间
	int length = bit_length;
	uint64_t  threadwrite_code = ((uint64_t)even_code << (24 + byte_restbits)) + ((uint64_t)odd_code << (24 + byte_restbits - even_code_size));
	int right_shift = 56;
	if (byte_restbits != 8) {
		write_bytelocation++;
		length -= byte_restbits;
		right_shift -= 8;
	}
	for (int i = length; i > 0; i = i - 8) {
		Write_JPEGdata[write_bytelocation] = (threadwrite_code >> right_shift) & 0XFF;
		right_shift -= 8;
		write_bytelocation++;
	}
	if (byte_restbits != 8) {
		if (bit_length < byte_restbits && bit_length)
			Write_JPEGdata[byte_location] = Write_JPEGdata[byte_location] | (threadwrite_code >> 56) & 0XFF;
		__syncthreads();
		if (bit_length >= byte_restbits)
			Write_JPEGdata[byte_location] = Write_JPEGdata[byte_location] | (threadwrite_code >> 56) & 0XFF;
	}
}

/***********************************************************************************************************
/***函数名称：CUDA_RGB2YUV_kernel
/***函数功能：将位图的BMP数据GRB模式转换为YUV数据模式
/***输    入：d_bsrc       原始的位图数据
/***输    入：nPitch       字节对齐的RGB数据大小
/***输    入：Size         字节对齐的YCrCb数据大小
/***输    出：Y\Cr\Cb      转换后的3个颜色分量
/***返    回：无返回
************************************************************************************************************/
__global__ void CUDA_RGB2YUV_kernel(BYTE *d_bsrc, BYTE *Y, BYTE *Cr, BYTE *Cb, size_t nPitch, size_t StrideF) {
	int tid = (blockIdx.x << 3) + threadIdx.x;
	d_bsrc += ((blockIdx.y << 3) + threadIdx.y) * nPitch + (tid << 1) + tid;
	int OffsThreadInRow = ((blockIdx.y << 3) + threadIdx.y) * StrideF + tid;

	float r = d_bsrc[2];
	float g = d_bsrc[1];
	float b = d_bsrc[0];
	Y[OffsThreadInRow] = (g * C_Yg + b * C_Yb + r * C_Yr);
	Cr[OffsThreadInRow] = (g * C_Ug + b * C_Ub + 128.f + r* C_Ur);
	Cb[OffsThreadInRow] = (g * C_Vg + b * C_Vb + 128.f + r* C_Vr);
}

/***********************************************************************************************************
/***函数名称：work_efficient_PrefixSum_kernel(int *X, int *BlockSum, int InputSize)
/***函数功能：前缀求和计算辅助函数，主要是数据块被分成n个小块以后，求每个小块的前缀和
/***输    入：X        需要进行前缀求和的数据
/***输    出：BlockSum  前缀求和每个小块的总和
/***输    出：X           前缀求和的数据的最终结果
/***返    回：无返回
************************************************************************************************************/
__global__ void work_efficient_PrefixSum_kernel(int *X, int *BlockSum) {
	// XY[2*BLOCK_SIZE] is in shared memory
	__shared__ int XY[512];
	__shared__ int XY1[512];
	int index;
	int tid = threadIdx.x << 1;
	int i = (blockIdx.x << 10) + tid + 1;
	XY[tid] = X[i];
	XY[tid + 1] = X[i] + X[i + 1];
	XY1[tid] = X[512 + i];
	XY1[tid + 1] = X[512 + i] + X[i + 513];
	__syncthreads();
	index = ((threadIdx.x + 1) << 2) - 1;
	if (index < 512) {
		XY[index] += XY[index - 2];
		XY1[index] += XY1[index - 2];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 3) - 1;
	if (index < 512) {
		XY[index] += XY[index - 4];
		XY1[index] += XY1[index - 4];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 4) - 1;
	if (index < 512) {
		XY[index] += XY[index - 8];
		XY1[index] += XY1[index - 8];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 5) - 1;
	if (index < 512) {
		XY[index] += XY[index - 16];
		XY1[index] += XY1[index - 16];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 6) - 1;
	if (index < 512) {
		XY[index] += XY[index - 32];
		XY1[index] += XY1[index - 32];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 7) - 1;
	if (index < 512) {
		XY[index] += XY[index - 64];
		XY1[index] += XY1[index - 64];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 8) - 1;
	if (index < 512) {
		XY[index] += XY[index - 128];
		XY1[index] += XY1[index - 128];
	}
	__syncthreads();
	if (index < 512) {
		XY[511] += XY[255];
		XY1[511] += XY1[255];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 8) - 1;
	if (index < 384) {
		XY[index + 128] += XY[index];
		XY1[index + 128] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 7) - 1;
	if (index < 448) {
		XY[index + 64] += XY[index];
		XY1[index + 64] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 6) - 1;
	if (index < 480) {
		XY[index + 32] += XY[index];
		XY1[index + 32] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 5) - 1;
	if (index < 496) {
		XY[index + 16] += XY[index];
		XY1[index + 16] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 4) - 1;
	if (index < 504) {
		XY[index + 8] += XY[index];
		XY1[index + 8] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 3) - 1;
	if (index < 508) {
		XY[index + 4] += XY[index];
		XY1[index + 4] += XY1[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 2) - 1;
	if (index < 510) {
		XY[index + 2] += XY[index];
		XY1[index + 2] += XY1[index];
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		X[1 + i] = XY[tid + 1];
		X[513 + i] = XY1[tid + 1];
	}
	else {
		X[i] = XY[tid] + XY[tid - 1];
		X[1 + i] = XY[tid + 1];
		X[512 + i] = XY1[tid] + XY1[tid - 1];
		X[513 + i] = XY1[tid + 1];
		BlockSum[(blockIdx.x << 1) + 1] = XY[511];
		BlockSum[(blockIdx.x << 1) + 2] = XY1[511];
	}
}

/***********************************************************************************************************
/***函数名称：work_efficient_BlockUp_kernel(int *dc_component)
/***函数功能：前缀求和计算辅助函数，主要是数据块被分成n个小块以后，求每个小块的前缀和
/***输    入：BlockSum    需要进行前缀求和的数据
/***输    出：BlockSum    前缀求和的数据的最终结果
/***返    回：无返回
************************************************************************************************************/
__global__ void work_efficient_BlockUp_kernel(int *BlockSum) {
	__shared__ int XY[512];
	int index;
	int tid = threadIdx.x << 1;
	int i = (blockIdx.x << 9) + tid + 1;
	XY[tid] = BlockSum[i];
	XY[tid + 1] = BlockSum[i] + BlockSum[i + 1];
	__syncthreads();
	index = ((threadIdx.x + 1) << 2) - 1;
	if (index < 512) {
		XY[index] += XY[index - 2];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 3) - 1;
	if (index < 512) {
		XY[index] += XY[index - 4];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 4) - 1;
	if (index < 512) {
		XY[index] += XY[index - 8];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 5) - 1;
	if (index < 512) {
		XY[index] += XY[index - 16];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 6) - 1;
	if (index < 512) {
		XY[index] += XY[index - 32];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 7) - 1;
	if (index < 512) {
		XY[index] += XY[index - 64];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 8) - 1;
	if (index < 512) {
		XY[index] += XY[index - 128];
	}
	__syncthreads();
	if (index < 512) {
		XY[511] += XY[255];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 8) - 1;
	if (index < 384) {
		XY[index + 128] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 7) - 1;
	if (index < 448) {
		XY[index + 64] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 6) - 1;
	if (index < 480) {
		XY[index + 32] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 5) - 1;
	if (index < 496) {
		XY[index + 16] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 4) - 1;
	if (index < 504) {
		XY[index + 8] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 3) - 1;
	if (index < 508) {
		XY[index + 4] += XY[index];
	}
	__syncthreads();
	index = ((threadIdx.x + 1) << 2) - 1;
	if (index < 510) {
		XY[index + 2] += XY[index];
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		BlockSum[1 + i] = XY[tid + 1];
	}
	else {
		BlockSum[i] = XY[tid] + XY[tid - 1];
		BlockSum[1 + i] = XY[tid + 1];
	}
}

__global__ void work_efficient_Adds_kernel(int *BlockSum, int *prefix_num) {
	int tid = (blockIdx.x << 9) + (threadIdx.x << 1) + 1;  //blockIdx.x*blockDim.x + threadIdx.x
	prefix_num[tid] = BlockSum[blockIdx.x] + prefix_num[tid];
	prefix_num[tid + 1] = BlockSum[blockIdx.x] + prefix_num[tid + 1];
}

/***********************************************************************************************************
/***函数名称：CUDA_DCT8_kernel
/***函数功能：对灰度原始图像数据进行DCT变换
/***输    入：X        需要进行前缀求和的数据
/***输    入：MCU_total   需要进行前缀求和的数据个数
/***输    出：X           前缀求和的数据的最终结果
/***返    回：无返回
************************************************************************************************************/
__global__ void CUDA_DCT8_kernel(BSI16 *d_ydst, BYTE *d_bsrc, RIM Size, int *DEV_ZIGZAG, float *DEV_STD_QUANT_TAB_LUMIN) {
	__shared__ float block[512];
	int OffsThreadInRow = (blockIdx.x << 6) + (threadIdx.z << 5) + (threadIdx.y << 3) + threadIdx.x;
	if (OffsThreadInRow >= Size.width) return;
	OffsThreadInRow = OffsThreadInRow - (blockIdx.x << 6);    //32*16中列偏移
	d_bsrc += ((blockIdx.y << 3) + threadIdx.x) * Size.StrideF + (blockIdx.x << 6) + (threadIdx.z << 5) + (threadIdx.y << 3);
	float *bl_ptr = block + (threadIdx.z << 5) + (threadIdx.y << 3) + (threadIdx.x << 6);

	float Vect0 = d_bsrc[0];
	float Vect1 = d_bsrc[1];
	float Vect2 = d_bsrc[2];
	float Vect3 = d_bsrc[3];
	float Vect4 = d_bsrc[4];
	float Vect5 = d_bsrc[5];
	float Vect6 = d_bsrc[6];
	float Vect7 = d_bsrc[7];

	float X07P = Vect0 + Vect7;
	float X16P = Vect1 + Vect6;
	float X25P = Vect2 + Vect5;
	float X34P = Vect3 + Vect4;

	float X07M = Vect0 - Vect7;
	float X61M = Vect6 - Vect1;
	float X25M = Vect2 - Vect5;
	float X43M = Vect4 - Vect3;

	float X07P34PP = X07P + X34P;
	float X07P34PM = X07P - X34P;
	float X16P25PP = X16P + X25P;
	float X16P25PM = X16P - X25P;

	bl_ptr[0] = X07P34PP + X16P25PP;
	bl_ptr[2] = C_b * X07P34PM + C_e * X16P25PM;
	bl_ptr[4] = X07P34PP - X16P25PP;
	bl_ptr[6] = C_e * X07P34PM - C_b * X16P25PM;

	bl_ptr[1] = C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M;
	bl_ptr[3] = C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M;
	bl_ptr[5] = C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M;
	bl_ptr[7] = C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M;
	bl_ptr = block + OffsThreadInRow;
	Vect0 = bl_ptr[0];
	Vect1 = bl_ptr[64];
	Vect2 = bl_ptr[128];
	Vect3 = bl_ptr[192];
	Vect4 = bl_ptr[256];
	Vect5 = bl_ptr[320];
	Vect6 = bl_ptr[384];
	Vect7 = bl_ptr[448];

	X07P = Vect0 + Vect7;
	X16P = Vect1 + Vect6;
	X25P = Vect2 + Vect5;
	X34P = Vect3 + Vect4;

	X07M = Vect0 - Vect7;
	X61M = Vect6 - Vect1;
	X25M = Vect2 - Vect5;
	X43M = Vect4 - Vect3;

	X07P34PP = X07P + X34P;
	X07P34PM = X07P - X34P;
	X16P25PP = X16P + X25P;
	X16P25PM = X16P - X25P;
	d_ydst = d_ydst + blockIdx.y * (Size.width << 3) + (blockIdx.x << 9) + (threadIdx.z << 8) + (threadIdx.y << 6);
	DEV_STD_QUANT_TAB_LUMIN += threadIdx.x;
	DEV_ZIGZAG += threadIdx.x;
	d_ydst[DEV_ZIGZAG[0]] = (X07P34PP + X16P25PP)* DEV_STD_QUANT_TAB_LUMIN[0];
	d_ydst[DEV_ZIGZAG[8]] = (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M) * DEV_STD_QUANT_TAB_LUMIN[8];
	d_ydst[DEV_ZIGZAG[16]] = (C_b * X07P34PM + C_e * X16P25PM) * DEV_STD_QUANT_TAB_LUMIN[16];
	d_ydst[DEV_ZIGZAG[24]] = (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M) * DEV_STD_QUANT_TAB_LUMIN[24];
	d_ydst[DEV_ZIGZAG[32]] = (X07P34PP - X16P25PP) * DEV_STD_QUANT_TAB_LUMIN[32];
	d_ydst[DEV_ZIGZAG[40]] = (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M) * DEV_STD_QUANT_TAB_LUMIN[40];
	d_ydst[DEV_ZIGZAG[48]] = (C_e * X07P34PM - C_b * X16P25PM) * DEV_STD_QUANT_TAB_LUMIN[48];
	d_ydst[DEV_ZIGZAG[56]] = (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M) * DEV_STD_QUANT_TAB_LUMIN[56];
}

/***********************************************************************************************************
/***函数名称：Data_codelength_kernel
/***函数功能：对扫描后数据进行编码，计算64个mcu bit流长度并进行scan扫描
/***int *dc_component 输入每个mcu的直流分量，并在kernel最后存储64个mcu bit流总长度
/***int *d_ydst  输入经过zigzag扫描后的数据
/***int *prefix_num用于在运算中存储每个mcu的bit流长度的前缀和
************************************************************************************************************/
__global__ void Data_codelength_kernel(BSI16 *d_ydst, int MCU_total, BYTE *d_JPEGdata,
	int *prefix_num, int offset, const int huffman_lut_offset) {
	//计算对应的图像block id号	
	const int block_idx = (blockIdx.y * gridDim.x << 2) + (blockIdx.x << 2) + threadIdx.y;
	if (block_idx >= MCU_total) return;

	__shared__ int Length_count[(THREAD_WARP + 1) * 4];
	d_ydst += block_idx << 6;
	const int load_idx = threadIdx.x * 2;
	int in_even = d_ydst[load_idx];
	const int in_odd = d_ydst[load_idx + 1];

	//对直流分量进行差分编码
	if (threadIdx.x == 0 && block_idx != 0) in_even = in_even - d_ydst[load_idx - 64];
	if (threadIdx.x == 0 && block_idx == 0) in_even = in_even - 85;

	//计算当前编码数据前面0的个数
	const unsigned int nonzero_mask = (1 << threadIdx.x) - 1;
	const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC数据都看作是非零数据
	const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
	const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
	const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);

	//计算当前线程偶编码数据编码前面0的个数
	int zeros_before_even = 2 * (zero_pair_count + threadIdx.x - 32);
	if ((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
		zeros_before_even += 1;
	}

	// true if any nonzero pixel follows thread's odd pixel
	const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;

	// 计算奇数位编码数据前面的编码 ,如果交流分量in_even是0，则in_odd前面的0的个数+1
	// (the count is actually multiplied by 16)
	int zeros_before_odd = (in_even || !threadIdx.x) ? 0 : zeros_before_even + 1;

	// clear zero counts if no nonzero pixel follows (so that no 16-zero symbols will be emited)
	// otherwise only trim extra bits from the counts of following zeros
	const int zero_count_mask = nonzero_follows ? 0xF : 0;
	zeros_before_even &= zero_count_mask;
	zeros_before_odd &= zero_count_mask;

	int even_lut_offset = huffman_lut_offset;
	if (0 == threadIdx.x) {
		// first thread uses DC part of the table for its even value
		even_lut_offset += 256 + 1;
	}

	// 一个block的结束标志
	if (0 == ((threadIdx.x ^ 31) | in_odd)) {
		// 如果需要添加结束标志，则将zeros_before_odd的值改为16
		zeros_before_odd = 16;
	}

	// each thread gets codeword for its two pixels
	unsigned int even_code = gpuhuffman_encode_value(zeros_before_even, in_even, even_lut_offset);
	unsigned int odd_code = gpuhuffman_encode_value(zeros_before_odd, in_odd, huffman_lut_offset);

	int *bl_ptr = Length_count + (THREAD_WARP + 1) * threadIdx.y;
	const unsigned int even_code_size = even_code & 31;
	const unsigned int odd_code_size = odd_code & 31;
	int bit_length = even_code_size + odd_code_size;
	even_code = even_code & ~31;
	odd_code = odd_code & ~31;
	int code_nbits = bit_length;

	//计算每个BLOCK中非零编码的数据个数
	unsigned int prefix_bitmap = __ballot(bit_length);
	int prefix_count = __popc(prefix_bitmap & nonzero_mask);
	if (bit_length) {
		bl_ptr[prefix_count] = bit_length;
		__syncthreads();
		//进行前缀求和运算
		for (int j = 0; j < prefix_count; j++) {
			code_nbits = code_nbits + bl_ptr[j];
		}
	}
	if (threadIdx.x == 31) {
		prefix_num[block_idx + 1] = code_nbits + 8;
	}
	//计算写入缓存区的具体字节位置，确定写入d_JPEGdata的位置
	BYTE *Write_JPEGdata = d_JPEGdata + (block_idx << 6);
	const int bit_location = code_nbits - bit_length;
	const int byte_restbits = (8 - (bit_location & MASK));
	const int byte_location = bit_location >> SHIFT;
	int write_bytelocation = byte_location;
	//将一个线程的数据编码写入数据编码缓存空间
	int length = bit_length;
	uint64_t  threadwrite_code = ((uint64_t)even_code << (24 + byte_restbits)) + ((uint64_t)odd_code << (24 + byte_restbits - even_code_size));
	int right_shift = 56;
	if (byte_restbits != 8) {
		write_bytelocation++;
		length -= byte_restbits;
		right_shift -= 8;
	}
	for (int i = length; i > 0; i = i - 8) {
		Write_JPEGdata[write_bytelocation] = (threadwrite_code >> right_shift) & 0XFF;
		right_shift -= 8;
		write_bytelocation++;
	}
	if (byte_restbits != 8) {
		if (bit_length < byte_restbits && bit_length)
			Write_JPEGdata[byte_location] = Write_JPEGdata[byte_location] | (threadwrite_code >> 56) & 0XFF;
		__syncthreads();
		if (bit_length >= byte_restbits)
			Write_JPEGdata[byte_location] = Write_JPEGdata[byte_location] | (threadwrite_code >> 56) & 0XFF;
	}
}

__global__ void CUDA_YCrCb_codelength_kernel(BSI16 *d_ydst, BYTE *d_JPEGdata, int *prefix_num, int MCU_total, int offset, int cycle) {
	int tid = (blockIdx.x << 7) + threadIdx.x;  //blockIdx.x*blockDim.x + threadIdx.x
	int bit_location = 0;
	if (tid >= MCU_total) return;
	int in_even, zeros_before = 0;
	d_ydst += tid << 6;

	//对直流分量和交流分量进行预处理
	if (tid == 0)
		in_even = d_ydst[0] - 85;
	else
		in_even = d_ydst[0] - d_ydst[-64];
	int in_odd = d_ydst[1];
	d_JPEGdata = d_JPEGdata + (tid << 6);

	unsigned int even_code = gpuhuffman_encode_value(0, in_even, (256 + 1) * 3);
	unsigned int odd_code = gpuhuffman_encode_value(0, in_odd, (256 + 1) * 2);
	unsigned int even_code_size = even_code & 31;
	unsigned int odd_code_size = odd_code & 31;
	int bit_length = even_code_size + odd_code_size;
	even_code = even_code & ~31;
	odd_code = odd_code & ~31;

	write_bitstream(even_code, odd_code, bit_length, bit_location, even_code_size, d_JPEGdata);
	bit_location += bit_length;
	for (int j = 2; j < cycle; j = j + 2) {
		in_even = d_ydst[j];
		in_odd = d_ydst[j + 1];
		if (!in_even) zeros_before++;
		odd_code = 0;
		even_code = 0;
		if (in_even)
			even_code = gpuhuffman_encode_value(zeros_before, in_even, (256 + 1) * 2);
		zeros_before = in_even ? 0 : zeros_before + 1;

		if (in_odd)
			odd_code = gpuhuffman_encode_value(zeros_before, in_odd, (256 + 1) * 2);
		if (in_even || in_odd) {
			even_code_size = even_code & 31;
			odd_code_size = odd_code & 31;
			bit_length = even_code_size + odd_code_size;
			even_code = even_code & ~31;
			odd_code = odd_code & ~31;
			write_bitstream(even_code, odd_code, bit_length, bit_location, even_code_size, d_JPEGdata);
			bit_location += bit_length;
		}
	}
	write_bitstream(0, 0, 2, bit_location, 0, d_JPEGdata);
	prefix_num[tid * 3 + offset] = bit_location + 2;
}

__global__ void adds_prefixsum(int *dc_component, int *prefix_num, int MCU_total) {
	int tid = (blockIdx.x << 7) + threadIdx.x;  //blockIdx.x*blockDim.x + threadIdx.x
	if (tid >= MCU_total) return;
	prefix_num[tid + 1] = dc_component[blockIdx.x] + prefix_num[tid + 1];
}

__global__ void adds_prefixsum1(int *dc_component, int *prefix_num, int MCU_total) {
	int tid = (blockIdx.x << 7) + threadIdx.x; //blockIdx.x*blockDim.x + threadIdx.x
	if (tid >= (MCU_total - 1) >> 7) return;
	prefix_num[tid + 1] = dc_component[blockIdx.x] + prefix_num[tid + 1];
}

__global__ void data_shift_kernel(BYTE *d_JPEGdata, int *prefix_num, int MCU_total, int *d_datalen, int *dc_component, int* last_prefix_num) {
	int tid = (blockIdx.x << 7) + threadIdx.x;  //blockIdx.x*blockDim.x + threadIdx.x
	int byte_location = 0;
	if (tid >= MCU_total) return;                                        //如果tid>MCU总数，则不执行
	d_JPEGdata = d_JPEGdata + (tid << 6);                                //计算之前编码好的数据流首地址
	BYTE *JPEG_Writedatalocation = d_JPEGdata + 63;                      //位移后的BYTE要写入的位置
	BYTE byte_tmp;
	int length = prefix_num[tid + 1] - prefix_num[tid] - 8;                //得到每个MCU编码数据bit流的所占的字节数
	int right_shift = prefix_num[tid] & MASK;                            //得到前个MCU编码数据bit流在本MCU编码数据bit流中首字节所占的bit数
	int left_shift = 8 - right_shift;                                    //得到本MCU编码数据bit流首字节所占的bit数
	byte_location = (length - 1) >> SHIFT;                                 //得到本MCU编码数据bit流尾字节所在位置
	int bit_rest = 8 - length + ((byte_location << SHIFT));
	length = length + right_shift + 8;                                   //得到本MCU编码数据bit流数据字节长度
	length >>= SHIFT;
	if (right_shift >= bit_rest) {
		JPEG_Writedatalocation[0] = (d_JPEGdata[byte_location] << left_shift);
		JPEG_Writedatalocation--;
	}
	for (; byte_location > 0; byte_location--) {
		byte_tmp = (d_JPEGdata[byte_location] >> right_shift) | (d_JPEGdata[byte_location - 1] << left_shift);
		if (byte_tmp == 0xff) {
			length++;
			JPEG_Writedatalocation[0] = 0;
			JPEG_Writedatalocation[-1] = byte_tmp;
			JPEG_Writedatalocation -= 2;
		}
		else {
			JPEG_Writedatalocation[0] = byte_tmp;
			JPEG_Writedatalocation--;
		}
	}

	byte_tmp = d_JPEGdata[0] >> right_shift;
	if (byte_tmp == 0xff) {
		length++;
		JPEG_Writedatalocation[0] = 0;
		JPEG_Writedatalocation--;
		JPEG_Writedatalocation[0] = byte_tmp;
	}
	else {
		JPEG_Writedatalocation[0] = byte_tmp;
	}
	last_prefix_num[tid + 1] = length;
}

__global__ void Data_encodelater1_kernel(int *prefix_num, BYTE *d_JPEGdata, BYTE *last_AC, int MCU_total, int *d_datalen)
{
	int tid = (blockIdx.x << 7) + threadIdx.x;   //blockIdx.x*blockDim.x + threadIdx.x
	if (tid >= MCU_total) return;
	int  length;
	if (tid == MCU_total - 1) d_datalen[0] = prefix_num[tid + 1];
	length = prefix_num[tid + 1] - prefix_num[tid];
	last_AC = last_AC + prefix_num[tid];
	d_JPEGdata = d_JPEGdata + (tid << 6) + 64 - length;

	for (int i = 0; i < length; i++)
	{
		last_AC[i] = d_JPEGdata[i];
	}
}

//-------------------------------------------------------结束----------------------------------------//

/*************************************************
函数名称: RmwRead8BitBmpFile2Img  //

函数描述: 函数将存储位置的.bmp格式图像读入内存中； //

输入参数：const char * filename ：输入图像文件路径；
.         unsigned char* pImg :存放24位位图的指针；
.		  unsigned char* Binarization :存放灰度图的指针；
.		  int* width :读出图像列数；
.		  int* width :读出图像行数；//

输出参数：unsigned char* pImg ：若输入图像为灰度图，则指针指向NULL。
.         unsigned char* Binarization ：若输入图像为24位彩图，则指针指向NULL。；//

返回值  : bool -- 读入成功标志位//

其他说明: 函数仅用于调试阶段，实际工程中相机采样照片已经存放在内存区域中；
.         该函数在调用前，需要先为图像指针分配图像大小的内存区域；
.         内存区域大小(Byte) =  width * height * ImgDeep；    //

*************************************************/
bool RmwRead8BitBmpFile2Img(const char * filename, unsigned char*pImg, unsigned char*Binarization, int *width, int *height)
{
	FILE *binFile;
	BITMAPFILEHEADER fileHeader;//文件头
	BITMAPINFOHEADER bmpHeader;//信息头
	BOOL isRead = TRUE;
	int ImgDeep;
	int linenum, ex; // nenum:一行像素的字节总数，包括填充字节

					 //open file
	if ((binFile = fopen(filename, "rb")) == NULL) return NULL;

	//read struts
	if (fread((void *)&fileHeader, 1, sizeof(fileHeader), binFile) != sizeof(fileHeader)) isRead = FALSE;
	if (fread((void *)&bmpHeader, 1, sizeof(bmpHeader), binFile) != sizeof(bmpHeader)) isRead = FALSE;

	if (isRead == FALSE || fileHeader.bfOffBits<sizeof(fileHeader) + sizeof(bmpHeader)) {
		fclose(binFile);
		return NULL;
	}

	//read image info
	*width = bmpHeader.biWidth;
	*height = bmpHeader.biHeight;
	ImgDeep = bmpHeader.biBitCount / 8;//每个像素所占字节数目
	linenum = (*width * ImgDeep + 3) / 4 * 4;//这里要改
	ex = linenum - *width * ImgDeep;   //每一行的填充字节

	fseek(binFile, fileHeader.bfOffBits, SEEK_SET);
	//读取灰度图
	if (ImgDeep == 1)
	{
		if (Binarization != NULL)
			for (int i = 0; i<*height; i++)
			{
				int r = fread(Binarization + (*height - i - 1)*(*width)*ImgDeep, sizeof(unsigned char), (*width)*ImgDeep, binFile);
				if (r != (*width)*ImgDeep)
				{
					delete Binarization;
					fclose(binFile);
					return NULL;
				}
				fseek(binFile, ex, SEEK_CUR);
			}
		fclose(binFile);
		return true;
	}
	//读取位图
	else if (ImgDeep == 3)
	{
		//pImg = new uchar[(*width)*(*height)*ImgDeep];
		if (pImg != NULL)
		{
			for (int i = 0; i < *height; i++)
			{
				int r = fread(pImg + (*height - i - 1)*(*width)*ImgDeep, sizeof(unsigned char), (*width)*ImgDeep, binFile);//**
				if (r != (*width)*ImgDeep)//**
				{
					fclose(binFile);
					return NULL;
				}
				fseek(binFile, ex, SEEK_CUR);
			}
			fclose(binFile);
			//bmp转灰度
			if (Binarization != NULL)
			{
				for (int i = 0; i < *height; i++)
					for (int j = 0; j < *width; j++)
					{
						Binarization[j + i * (*width)] = pImg[j * ImgDeep + i * (*width) * ImgDeep] * 0.299 +
							pImg[j * ImgDeep + 1 + i * (*width) * ImgDeep] * 0.587 +
							pImg[j * ImgDeep + 2 + i * (*width) * ImgDeep] * 0.114;
					}
			}
			return true;
		}
		else//
		{
			unsigned char *tempImg = new uchar[(*width)*(*height)*ImgDeep];
			if (tempImg != NULL)
			{
				for (int i = 0; i < *height; i++)
				{
					int r = fread(tempImg + (*height - i - 1)*(*width)*ImgDeep, sizeof(unsigned char), (*width)*ImgDeep, binFile);//**
					if (r != (*width)*ImgDeep)//**
					{
						delete[]tempImg;
						fclose(binFile);
						return NULL;
					}
					fseek(binFile, ex, SEEK_CUR);
				}
				fclose(binFile);
				//bmp转灰度
				if (Binarization != NULL)
				{
					for (int i = 0; i < *height; i++)
						for (int j = 0; j < *width; j++)
						{
							Binarization[j + i * (*width)] = tempImg[j * ImgDeep + i * (*width) * ImgDeep] * 0.299 +
								tempImg[j * ImgDeep + 1 + i * (*width) * ImgDeep] * 0.587 +
								tempImg[j * ImgDeep + 2 + i * (*width) * ImgDeep] * 0.114;
						}
				}
				delete[]tempImg;
				return true;
			}
		}
	}
	else return false;
}

/*************************************************
函数名称: RmwWrite8bitImg2BmpFile  //

函数描述: 函数将内存位置的.bmp格式图像写入到硬盘中； //

输入参数：unsigned char* pImg :存放灰度图的指针；
.		  int* width :图像列数；
.		  int* width :图像行数；
.		  const char * filename ：输出图像文件路径；//

输出参数：const char * filename ：.bmp格式灰度图。；//

返回值  : Suc(bool型) -- 写出成功标志位    //

其他说明: 函数仅用于调试阶段，实际工程中相机采样照片已经存放在内存区域中；
.         该函数在调用前，需要先为图像指针分配图像大小的内存区域；
.         内存区域大小(Byte) =  width * height * ImgDeep；    //

*************************************************/
bool RmwWrite8bitImg2BmpFile(unsigned char *pImg, int width, int height, const char * filename)
{
	FILE * BinFile;
	BITMAPFILEHEADER FileHeader;
	BITMAPINFOHEADER BmpHeader;
	int i, extend;
	bool Suc = true;
	unsigned char p[4], *pCur;
	unsigned char* ex;

	extend = (width + 3) / 4 * 4 - width;

	// Open File
	if ((BinFile = fopen(filename, "w+b")) == NULL) { return false; }
	//参数填法见结构链接
	FileHeader.bfType = ((WORD)('M' << 8) | 'B');
	FileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * 4L;//2个头结构后加调色板
	FileHeader.bfSize = FileHeader.bfOffBits + (width + extend)*height;
	FileHeader.bfReserved1 = 0;
	FileHeader.bfReserved2 = 0;
	if (fwrite((void *)&FileHeader, 1, sizeof(FileHeader), BinFile) != sizeof(FileHeader)) Suc = false;
	// Fill the ImgHeader
	BmpHeader.biSize = 40;
	BmpHeader.biWidth = width;
	BmpHeader.biHeight = height;
	BmpHeader.biPlanes = 1;
	BmpHeader.biBitCount = 8;
	BmpHeader.biCompression = 0;
	BmpHeader.biSizeImage = 0;
	BmpHeader.biXPelsPerMeter = 0;
	BmpHeader.biYPelsPerMeter = 0;
	BmpHeader.biClrUsed = 0;
	BmpHeader.biClrImportant = 0;
	if (fwrite((void *)&BmpHeader, 1, sizeof(BmpHeader), BinFile) != sizeof(BmpHeader)) Suc = false;
	// 写入调色板
	for (i = 0, p[3] = 0; i<256; i++)
	{
		p[0] = p[1] = p[2] = i; // blue,green,red; //改255 - i则灰度反转
		if (fwrite((void *)p, 1, 4, BinFile) != 4) { Suc = false; break; }
	}

	if (extend)
	{
		ex = new unsigned char[extend]; //填充数组大小为 0~3
		memset(ex, 0, extend);
	}

	//write data
	for (pCur = pImg + (height - 1)*width; pCur >= pImg; pCur -= width)
	{
		if (fwrite((void *)pCur, 1, width, BinFile) != (unsigned int)width) Suc = false; // 真实的数据
		if (extend) // 扩充的数据 这里填充0
			if (fwrite((void *)ex, 1, extend, BinFile) != 1) Suc = false;
	}

	// return;
	fclose(BinFile);
	if (extend)
		delete[] ex;
	return Suc;
}

/*************************************************
函数名称: GetImgBoxHost  //

函数描述: 预提取包围盒函数。在矩形模式时需要预先好提取包围盒。函数利用CPU版本的八邻域追踪法提取输入图像的包围盒。
.		  提取出的包围盒数据保存在全局变量vector<RecData>gHostRecData中。
.         函数初始化了包围盒更新相关的全局变量//

输入参数：const char * filename -所要提取的位图(*.bmp)的绝对路径            //

输出参数：无   //

返回值  : 无  //

其他说明: 函数将提取的包围盒数据保存在全局变量vector<RecData>gHostRecData中，并且对容器 gHostRecData中元素数目进行了规整,
.         在容器末尾添加元素0，将容器数目填充为了128的整数倍//

*************************************************/
void GetImgBoxHost(const char *path)
{
	Parameter devpar;
	//初始化图像信息参数
	devpar.ImgHeight = gStructVarible.ImgHeight;
	devpar.ImgWidth = gStructVarible.ImgWidth;
	devpar.Threshold = gStructVarible.Threshold;
	devpar.LengthMin = gStructVarible.LengthMin;
	devpar.LengthMax = gStructVarible.LengthMax;
	devpar.AreaMin = gStructVarible.AreaMin;
	devpar.AreaMax = gStructVarible.AreaMax;
	devpar.PictureNum = gStructVarible.PictureNum;
	devpar.RecPadding = gStructVarible.RecPadding;
	//方位数组申明
	const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
	//初始化CPU端方位盒数据
	if (gHostRecData.size() != 0)
		gHostRecData.clear();
	//图像空间分配
	unsigned char *ImgHostdata = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum]; //qwt这里程序有BUG
	unsigned char *m_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//二值化图
	unsigned char *n_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//膨胀图
	unsigned char *c_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//轮廓图	
	unsigned char *temp_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//临时变量图
																									 //读取图片
	int Picoffset = devpar.ImgHeight * devpar.ImgWidth;
	for (int j = 0; j < devpar.PictureNum; j++)
	{
		RmwRead8BitBmpFile2Img(path, NULL, ImgHostdata + j*Picoffset, &devpar.ImgWidth, &devpar.ImgHeight);
	}
	//二值化
	for (int i = 0; i <devpar.ImgHeight*devpar.PictureNum; i++)
	{
		for (int j = 0; j < devpar.ImgWidth; j++)
		{
			m_ptr[j + i * devpar.ImgWidth] = ImgHostdata[j + i * devpar.ImgWidth] > devpar.Threshold ? 255 : 0;
			c_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
			n_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
			temp_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
		}

	}
	//膨胀
	for (int i = 1; i<devpar.ImgHeight*devpar.PictureNum - 1; i++)
		for (int j = 1; j <devpar.ImgWidth - 1; j++)
		{
			if (m_ptr[j + i * devpar.ImgWidth] == 0)
			{
				if (m_ptr[j - 1 + (i - 1)*devpar.ImgWidth] != 0 || m_ptr[j + (i - 1)*devpar.ImgWidth] != 0 || m_ptr[j + 1 + (i - 1)*devpar.ImgWidth] != 0 ||
					m_ptr[j - 1 + i * devpar.ImgWidth] != 0 || m_ptr[j + 1 + i * devpar.ImgWidth] != 0 ||
					m_ptr[j - 1 + (i + 1)*devpar.ImgWidth] != 0 || m_ptr[j + (i + 1)*devpar.ImgWidth] != 0 || m_ptr[j + 1 + (i + 1)*devpar.ImgWidth] != 0)
				{
					n_ptr[j + i * devpar.ImgWidth] = 255;
					c_ptr[j + i * devpar.ImgWidth] = 255;
					temp_ptr[j + i * devpar.ImgWidth] = 255;
				}
			}
		}
	//腐蚀  c_ptr是轮廓
	for (int i = 1; i<devpar.ImgHeight*devpar.PictureNum - 1; i++)
		for (int j = 1; j < devpar.ImgWidth - 1; j++)
		{
			if (n_ptr[j + i * devpar.ImgWidth] != 0)
			{
				if (n_ptr[j + (i - 1)*devpar.ImgWidth] != 0 && n_ptr[j - 1 + i * devpar.ImgWidth] != 0 &&
					n_ptr[j + 1 + i * devpar.ImgWidth] != 0 && n_ptr[j + (i + 1)*devpar.ImgWidth] != 0)
				{
					c_ptr[j + i * devpar.ImgWidth] = 0;
					temp_ptr[j + i * devpar.ImgWidth] = 0;
				}
			}
		}
	//方位盒
	short xmax;
	short xmin;
	short ymax;
	short ymin;
	// 边缘跟踪  
	int i, j, counts = 0, curr_d = 0;//counts用于循环计数   curr_d是方向数组的索引ID
	short cLength;
	//提取方位盒子
	for (i = 1; i <devpar.ImgHeight*devpar.PictureNum - 1; i++)
		for (j = 1; j <devpar.ImgWidth - 1; j++)
		{
			// 起始点及当前点  
			cv::Point b_pt = cv::Point(i, j);
			cv::Point c_pt = cv::Point(i, j);
			// 如果当前点为前景点  
			if (255 == c_ptr[j + i * devpar.ImgWidth])
			{
				cLength = 1;
				xmin = xmax = i;
				ymin = ymax = j;

				bool tra_flag = false;//设置标志位
				c_ptr[j + i * devpar.ImgWidth] = 0;// 用过的点直接给设置为0  
												   // 进行跟踪  
				while (!tra_flag)
				{
					// 循环八次  
					for (counts = 0; counts < 8; counts++)
					{
						// 防止索引出界  
						if (curr_d >= 8)
						{
							curr_d -= 8;
						}
						if (curr_d < 0)
						{
							curr_d += 8;
						}
						// 跟踪的过程，是个连续的过程，需要不停的更新搜索的root点  
						c_pt = cv::Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);
						// 边界判断  
						if ((c_pt.x > 0) && (c_pt.x < devpar.ImgHeight*devpar.PictureNum - 1) &&
							(c_pt.y > 0) && (c_pt.y < devpar.ImgWidth - 1))
						{
							// 如果存在边缘  
							if (255 == c_ptr[c_pt.x*devpar.ImgWidth + c_pt.y])
							{
								//更新包围盒
								xmax = xmax > c_pt.x ? xmax : c_pt.x;
								ymax = ymax > c_pt.y ? ymax : c_pt.y;
								xmin = xmin < c_pt.x ? xmin : c_pt.x;
								ymin = ymin < c_pt.y ? ymin : c_pt.y;
								curr_d -= 2;   //更新当前方向  
								c_ptr[c_pt.x*devpar.ImgWidth + c_pt.y] = 0;
								// 更新b_pt:跟踪的root点  
								b_pt.x = c_pt.x;
								b_pt.y = c_pt.y;
								cLength++;
								break;   // 跳出for循环  
							}
						}
						curr_d++;
					}   // end for  
						// 跟踪的终止条件：如果8邻域都不存在边缘  
					if (8 == counts)
					{
						// 清零  
						curr_d = 0;
						tra_flag = true;
						if (cLength < devpar.LengthMax && (cLength > devpar.LengthMin))
						{
							RecData tempRecData;
							int tempcount = 0;
							if (0.7<double(xmax - xmin) / double(ymax - ymin) < 1.5)//高/宽
							{

								//轮廓图中心点9领域判断
								for (int k = -1; k < 2; k++)
								{
									if ((xmax + xmax) / 2 < devpar.ImgHeight*devpar.PictureNum && (ymax + ymin) / 2 < devpar.ImgWidth)
									{
										tempcount += temp_ptr[(ymax + ymin) / 2 - 1 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
										tempcount += temp_ptr[(ymax + ymin) / 2 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
										tempcount += temp_ptr[(ymax + ymin) / 2 + 1 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
									}
								}
								//轮廓横纵向-边判断
								for (int k = xmin; k <= xmax; k++)//判断Height方向
								{
									tempcount += temp_ptr[(ymax + ymin) / 2 + k*devpar.ImgWidth] > 0 ? 1 : 0;
								}
								for (int k = ymin; k <= ymax; k++)//判断width方向
								{
									tempcount += temp_ptr[k + (xmax + xmin) / 2 * devpar.ImgWidth] > 0 ? 1 : 0;
								}
								if (tempcount <= 4)
								{
									if (xmin - devpar.RecPadding < 0)
										tempRecData.RecXmin = 0;
									else
										tempRecData.RecXmin = xmin - devpar.RecPadding;
									if (ymin - devpar.RecPadding < 0)
										tempRecData.RecYmin = 0;
									else
										tempRecData.RecYmin = ymin - devpar.RecPadding;
									if (xmax + devpar.RecPadding > devpar.ImgHeight*devpar.PictureNum - 1)
										tempRecData.RecXmax = devpar.ImgHeight*devpar.PictureNum - 1;
									else
										tempRecData.RecXmax = xmax + devpar.RecPadding;
									if (ymax + devpar.RecPadding > devpar.ImgWidth)
										tempRecData.RecYmax = devpar.ImgWidth - 1;
									else
										tempRecData.RecYmax = ymax + devpar.RecPadding;
									gHostRecData.push_back(tempRecData);
								}
							}
						}
						break;
					}
				}  // end if  
			}  // end while  
		}
	//规整方位盒数量，利用后续线程配置
	gSingleImgRecNum = gHostRecData.size() / devpar.PictureNum;//这是单张图方位盒的实际数量
	int rRecNum = (gHostRecData.size() + 127) / 128 * 128;
	gHostRecData.resize(rRecNum, RecData{ 0,0,0,0 });
	gRecNum = rRecNum;//包围盒数量
					  //释放内存
	delete[]ImgHostdata;
	delete[]m_ptr;
	delete[]n_ptr;
	delete[]c_ptr;
	delete[]temp_ptr;
}

//-----------------------------------------功能处理类---------------------------------------//
//--------------------------------------------开始------------------------------------------//
/*----------------------------------全图模式标志点提取处理类------------------------------*/
class SIM : public Runnable
{
public:
	HardwareInfo HardwarePar;//硬件参数
	Parameter Devpar;//图像参数
	~SIM()//析构函数
	{
	}
	void Run()
	{
		//设置GPU设备号
		cudaSetDevice(HardwarePar.GpuId);
		//调试项
		cudaError_t  err, err1;
		clock_t start, end;
		clock_t startp, overp;
		clock_t time3;
		/*获取当前线程号*/

		/***********/
		int img_index;
		char DataFilename[100];
		char strFilename[100];
		const char* path = Devpar.DataReadPath;
		int OutPutInitialIndex = 0; //输出的Bin文件初始索引号
		int BufferIndex = 0;//页锁缓冲区索引
		long long  Bufferoffset = 0;//缓冲区偏移量
		bool DatafullFlag = false;//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。

		/*----------------------参数计算------------------------------------------*/
		Devpar.ImgChannelNum = Devpar.ImgBitDeep / 8;//位深转换成通道数
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//填充后的宽度计算
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / Devpar.PicBlockSize;
		Devpar.ColThreadNum = (Devpar.ImgWidth / Devpar.PicBlockSize + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);


		/*----------------------内存申请------------------------------------------*/
		//创建CUDA流
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/****  图像数据  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//设备内存
		unsigned char* DevPadding[CUDAStreams];//填充边界后的图像内存   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//二值化图
		unsigned char* DevCounter[CUDAStreams];//轮廓图，在执行findcountores之后才生成
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight * Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char));  //qwt7.26
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/*主机端*/
		//输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*设备端*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//申请的临时变量空间，包括有方位盒、输出特征的GPU端内存和GPU显存
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//输出周长
			cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//面积
			cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//重心坐标x
			cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//重心坐标y
			cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//特征索引号
			cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//方位盒 xmin
			cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//	    ymin
			cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//		xmax
			cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//		ymax
			cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//设备端输出	周长
			cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//				面积
			cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double) * 2);//				xpos
			cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double) * 2);//				ypos
			err = cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//				索引号
		}

		//标志点提取完整流程
		while ((img_index + CUDAStreams) <= gHostPathImgNumber && gStructVarible.TerminateFlag == 0)
		{
			//若图像类型为灰度图-即单通道，则直接将数据拷贝到DevPicGray
			if (Devpar.ImgChannelNum == 1)
			{
				for (int i = 0; i < CUDAStreams; i++)
				{
					Bufferoffset = long long(img_index + i)* Devpar.ImgHeight * Devpar.ImgWidth;
					cudaMemcpyAsync(DevPicGray[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//执行灰度化，二值化核函数程序
					GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
				}
			}
			else if (Devpar.ImgChannelNum == 3)//若图像类型为彩色图-即多通道，则直接将数据拷贝到DevPicColor
			{
				for (int i = 0; i < CUDAStreams; i++)
				{
					Bufferoffset = long long(img_index + i)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
					cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)//转灰度+padding
				{
					ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
				}
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//边界提取
				Dilation << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(Dev2Val[i], DevCounter[i], sizeof(uchar)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				Erosion << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取轮廓和边缘盒
				GetCounter << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], Devpar);//提取轮廓的函数
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				GetNonRepeatBox << <mGrid2, 128, 0, CStreams[i] >> > (gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取面积和重心//提取特征信息核函数
				GetInfo << <mGrid2, 128, 0, CStreams[i] >> > (DevPadding[i], gpDevIndex[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(double)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(double)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostIndex[i], gpDevIndex[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				err = cudaStreamSynchronize(CStreams[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选打印提取的特征
				vector<CircleInfo>myInfo;
				for (int j = 0; j < Devpar.ColThreadNum * Devpar.RowThreadNum; j++)
				{
					if (gpHostIndex[i][j] != 0)
					{
						CircleInfo temp;
						temp.index = (short)j;
						temp.length = gpHostLength[i][j];
						temp.area = gpHostArea[i][j];
						temp.xpos = gpHostXpos[i][j];
						temp.ypos = gpHostYpos[i][j];
						myInfo.push_back(temp);
					}
				}
				SignPoint.PointNumbers = myInfo.size();
				//输出标志点数据
				if (myInfo.size() > 0)
				{
					FILE* fp;
					sprintf_s(DataFilename, "%s\\%d.bin", Devpar.DataReadPath, img_index + HardwarePar.DeviceID * HardwarePar.CUDAStreamNum + i + 1); //【3】将图片的路径名动态的写入到DataFilename这个地址的内存空间
					fp = fopen(DataFilename, "wb");
					fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
					fclose(fp);
				}
			}
			img_index += HardwarePar.DeviceCount * HardwarePar.CUDAStreamNum;
		}

		/****       测试用代码       ****/
		/****  用于测试手动停止位置  ****/
		if (gStructVarible.TerminateFlag == 1)
		{
			char buffer[20];
			sprintf_s(buffer, "%s%d", "img_index = ", img_index);
			FILE* fp;
			sprintf_s(DataFilename, "%s\\%d.txt", Devpar.DataReadPath, 0); //【3】将图片的路径名动态的写入到DataFilename这个地址的内存空间
			fp = fopen(DataFilename, "wb");
			fwrite(buffer, sizeof(char) * 20, 1, fp);
			fclose(fp);
		}
		/**********************/

		//释放内存
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaFree(DevPicColor[i]);
			cudaFree(DevPicGray[i]);
			cudaFree(DevPadding[i]);
			cudaFree(Dev2Val[i]);
			cudaFree(DevCounter[i]);

			cudaFreeHost(gpHostLength[i]);
			cudaFreeHost(gpHostArea[i]);
			cudaFreeHost(gpHostXpos[i]);
			cudaFreeHost(gpHostYpos[i]);
			cudaFreeHost(gpHostIndex[i]);
			//设备端内存
			cudaFree(gpDevRecXLeft[i]);
			cudaFree(gpDevRecYLeft[i]);
			cudaFree(gpDevRecXRight[i]);
			cudaFree(gpDevRecYRight[i]);
			cudaFree(gpDevLength[i]);
			cudaFree(gpDevArea[i]);
			cudaFree(gpDevXpos[i]);
			cudaFree(gpDevYpos[i]);
			cudaFree(gpDevIndex[i]);
			cudaStreamDestroy(CStreams[i]);
		}
	}
};

class R : public Runnable
{
public:
	Parameter Devpar;//变量传参
	HardwareInfo HardwarePar;//硬件参数
	static int  mRindex;
	~R()
	{
	}
	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	void Run()
	{		
		//设置GPU设备号
		cudaSetDevice(HardwarePar.GpuId);
		//调试项
		/***********/
		int img_index;
		char strFilename[100];
		const char* path = Devpar.DataReadPath;
		int OutPutInitialIndex = 0; //输出的Bin文件初始索引号
		int BufferIndex = 0;//页锁缓冲区索引
		long long  Bufferoffset = 0;//缓冲区偏移量
		bool DatafullFlag = false;//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。

		/*----------------------参数计算------------------------------------------*/
		Devpar.ImgChannelNum =  Devpar.ImgBitDeep / 8;//位深转换成通道数
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//填充后的宽度计算
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / Devpar.PicBlockSize;//这里可能会有BUG-当高度不是PicBlock的整数倍时，可能出现问题
		Devpar.ColThreadNum = (Devpar.ImgWidth / Devpar.PicBlockSize + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);


		/*----------------------内存申请------------------------------------------*/
		//创建CUDA流
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/****  图像数据  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//设备内存
		unsigned char* DevPadding[CUDAStreams];//填充边界后的图像内存   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//二值化图
		unsigned char* DevCounter[CUDAStreams];//轮廓图，在执行findcountores之后才生成
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight * Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char)); 
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/*主机端*/
		//输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*设备端*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//申请的临时变量空间，包括有方位盒、输出特征的GPU端内存和GPU显存
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//输出周长
			cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//面积
			cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//重心坐标x
			cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//重心坐标y
			cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//特征索引号
			cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//方位盒 xmin
			cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
			cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
			cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
			cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//设备端输出	周长
			cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				面积
			cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				xpos
			cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				ypos
			cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				索引号
		}

		ExtractPointInitialSuccessFlag[HardwarePar.DeviceID] = true;

		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			vector<CircleInfo>myInfo;
			img_index = 0;//图像计数
			Bufferoffset = 0;//页锁内存偏移
			
            //绑定缓冲区
			while (true)
			{
				gExtrackPointLock.lock();
				mRindex = mRindex % (HardwareParam.DeviceCount + 1);
				if (PageLockBufferEmpty[mRindex] == false && PageLockBufferWorking[mRindex] == false)
				{
					PageLockBufferWorking[mRindex] = true;//将页锁内存标志位置为工作状态--进行绑定
					OutPutInitialIndex = PageLockBufferStartIndex[mRindex] * Bufferlength;//获取图像首索引
					BufferIndex = mRindex;
					DatafullFlag = true;
					mRindex++;
					gExtrackPointLock.unlock();
					break;
				}
				mRindex++;
				gExtrackPointLock.unlock();
				if (ExtractPointSuccess)
					break;
			}
			//处理数据
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength) //qwt
				{
					gExtrackPointLock.lock();
					PageLockBufferWorking[BufferIndex] = false;//处理结束--working标志位置为false
					gExtrackPointLock.unlock();
					PageLockBufferEmpty[BufferIndex] = true;  //
					DatafullFlag = false;
					break;
				}
				//若图像类型为灰度图-即单通道，则直接将数据拷贝到DevPicGray
				if (Devpar.ImgChannelNum == 1)
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)* Devpar.ImgHeight * Devpar.ImgWidth;
						cudaMemcpyAsync(DevPicGray[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)
					{
						//执行灰度化，二值化核函数程序
						GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
					}
				}
				else if (Devpar.ImgChannelNum == 3)//若图像类型为彩色图-即多通道，则直接将数据拷贝到DevPicColor
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
						cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + +Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)//转灰度+padding
					{
						ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
					}
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//执行灰度化，二值化核函数程序
					Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//边界提取
					Dilation << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(Dev2Val[i], DevCounter[i], sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					Erosion << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//提取轮廓和边缘盒
					GetCounter << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], Devpar);//提取轮廓的函数
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//筛选提取出的特征数组的非重复信息
					SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//筛选提取出的特征数组的非重复信息
					SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//筛选提取出的特征数组的非重复信息
					GetNonRepeatBox << <mGrid2, 128, 0, CStreams[i] >> > (gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//提取面积和重心//提取特征信息核函数
					GetInfo << <mGrid2, 128, 0, CStreams[i] >> > (DevPadding[i], gpDevIndex[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(double)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(double)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostIndex[i], gpDevIndex[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaStreamSynchronize(CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					for (int k = 0; k < Devpar.PictureNum; k++)
					{
						int hostindex = 0;
						int headpos = myInfo.size();
						CircleInfo headInfo;
						headInfo.index = OutPutInitialIndex + img_index + i;//对应文件索引
						headInfo.xpos = 99999;
						headInfo.ypos = 99999;//xpos 和 ypos作为头标志位
						headInfo.area = 0;	  //area为0也作为特征标志位
						myInfo.push_back(headInfo);
						for (int j = k*Devpar.ColThreadNum * Devpar.RowThreadNum / Devpar.PictureNum; j < (k + 1)*Devpar.ColThreadNum * Devpar.RowThreadNum / Devpar.PictureNum; j++)
						{
							if (gpHostIndex[i][j] != 0)
							{
								hostindex++;
								CircleInfo temp;
								temp.index = (short)hostindex;
								temp.length = gpHostLength[i][j];
								temp.area = gpHostArea[i][j];
								temp.xpos = gpHostXpos[i][j];
								temp.ypos = gpHostYpos[i][j];
								myInfo.push_back(temp);
							}
						}
						myInfo[headpos].length = hostindex;//长度置位
					}
				}
				img_index += HardwarePar.CUDAStreamNum*Devpar.PictureNum;
			}
			//	写磁盘
			if (myInfo.size() > 0)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", path, OutPutInitialIndex); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
				fp = fopen(strFilename, "wb");
				fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
				fclose(fp);
			}
		}
		for (int i = 0; i < CUDAStreams; i++)
		{			
			cudaFree(DevPicColor[i]);
			cudaFree(DevPicGray[i]);
			cudaFree(DevPadding[i]);
			cudaFree(Dev2Val[i]);
			cudaFree(DevCounter[i]);

			cudaFreeHost(gpHostLength[i]);
			cudaFreeHost(gpHostArea[i]);
			cudaFreeHost(gpHostXpos[i]);
			cudaFreeHost(gpHostYpos[i]);
			cudaFreeHost(gpHostIndex[i]);
			//设备端内存
			cudaFree(gpDevRecXLeft[i]);
			cudaFree(gpDevRecYLeft[i]);
			cudaFree(gpDevRecXRight[i]);
			cudaFree(gpDevRecYRight[i]);
			cudaFree(gpDevLength[i]);
			cudaFree(gpDevArea[i]);
			cudaFree(gpDevXpos[i]);
			cudaFree(gpDevYpos[i]);
			cudaFree(gpDevIndex[i]);
			cudaStreamDestroy(CStreams[i]);
		}
	}
};
int R::mRindex = 0;//静态变量初始化

/*----------------------------------矩形模式标志点提取处理类------------------------------*/
class RecR : public Runnable
{
public:
	HardwareInfo HardwarePar;//硬件参数
	Parameter Devpar;//变量传参	
	static int  mRecindex;
public:
	~RecR()//析构函数
	{
	}
	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	void Run()
	{

		//设置GPU设备号
		cudaSetDevice(HardwarePar.GpuId);
		//变量申明
		char DataFilename[100]; //定义一个字符数组保存----图片的读取路径 
		int img_index = 0;//输出图像 bin索引
		int OutPutInitialIndex = 0; //输出的Bin文件初始索引号
		int BufferIndex = 0;//页锁缓冲区索引
		long long  Bufferoffset = 0;//缓冲区偏移量
		bool DatafullFlag = false;//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。
		const char* path = Devpar.DataReadPath;

		/*----------------------参数计算------------------------------------------*/
		Devpar.ImgChannelNum = Devpar.ImgBitDeep / 8;//位深转换成通道数
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//填充后的宽度计算
		int Gridsize = gRecNum / 128;
		if (Gridsize == 0)//qwt823
			Gridsize = 1;
		/****  核函数Grid  ****/
		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Gridsize, 1, 1);

		/*----------------------存储区空间申请------------------------------------------*/
		//创建CUDA流
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/***  图像数据  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//设备内存
		unsigned char* DevPadding[CUDAStreams];//填充边界后的图像内存   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//二值化图
		unsigned char* DevCounter[CUDAStreams];//轮廓图，在执行findcountores之后才生成
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char));  //qwt7.26
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/****  主机端  ****/
		//标志点信息输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];

		/****  设备端  ****/
		//标志点信息输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		RecData *gpRDevRecData[CUDAStreams];//qwt821
	    //拷贝方位盒数据
		if (gRecNum > 0)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMalloc((void**)&gpRDevRecData[i], gRecNum * sizeof(RecData) * 2);//这里这个2的作用是：方位盒可能在实验期间数目要变，可能会变多一点，防止变了之后内存越界
				cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], gRecNum * sizeof(RecData), cudaMemcpyHostToDevice);
			}
		}

		//申请的临时变量空间，包括有方位盒、输出特征的GPU端内存和GPU显存
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], gRecNum * sizeof(short), cudaHostAllocDefault);//输出周长
			cudaHostAlloc((void**)&gpHostArea[i], gRecNum * sizeof(short), cudaHostAllocDefault);//面积
			cudaHostAlloc((void**)&gpHostXpos[i], gRecNum * sizeof(double), cudaHostAllocDefault);//重心坐标x
			cudaHostAlloc((void**)&gpHostYpos[i], gRecNum * sizeof(double), cudaHostAllocDefault);//重心坐标y
			cudaMalloc((void**)&gpDevLength[i], gRecNum * sizeof(short));//设备端输出	周长
			cudaMalloc((void**)&gpDevArea[i], gRecNum * sizeof(short));//				面积
			cudaMalloc((void**)&gpDevXpos[i], gRecNum * sizeof(double));//				xpos
			cudaMalloc((void**)&gpDevYpos[i], gRecNum * sizeof(double));//				ypos
		}

		ExtractPointInitialSuccessFlag[HardwarePar.DeviceID] = true;

		//标志点提取完整流程
		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			vector<CircleInfo>myInfo;
			img_index = 0;//图像计数
			Bufferoffset = 0;//页锁内存偏移
							 //绑定数据
			while (true)
			{
				gExtrackPointLock.lock();
				mRecindex = mRecindex % (HardwareParam.DeviceCount + 1);
				if (PageLockBufferEmpty[mRecindex] == false && PageLockBufferWorking[mRecindex] == false)
				{
					PageLockBufferWorking[mRecindex] = true;//将页锁内存标志位置为工作状态--进行绑定
					OutPutInitialIndex = PageLockBufferStartIndex[mRecindex] * Bufferlength;//获取图像首索引
					BufferIndex = mRecindex;
					DatafullFlag = true;
					mRecindex++;
					gExtrackPointLock.unlock();
					break;
				}
				mRecindex++;
				gExtrackPointLock.unlock();
				if (ExtractPointSuccess)
					break;
			}
			//提取特征
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength) //qwt
				{
					gExtrackPointLock.lock();
					PageLockBufferWorking[BufferIndex] = false;//处理结束--working标志位置为false
					gExtrackPointLock.unlock();
					PageLockBufferEmpty[BufferIndex] = true;  //
					DatafullFlag = false;
					break;
				}
				if (Devpar.ImgChannelNum == 1)
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)* Devpar.ImgHeight * Devpar.ImgWidth;
						cudaMemcpyAsync(DevPicGray[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)
					{
						//执行灰度化，二值化核函数程序
						GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
					}
				}
				else if (Devpar.ImgChannelNum == 3)//若图像类型为彩色图-即多通道，则直接将数据拷贝到DevPicColor
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
						cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + +Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)//转灰度+padding
					{
						ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
					}
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//执行灰度化，二值化核函数程序
					Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//边界提取
					Dilation << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(Dev2Val[i], DevCounter[i], sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					Erosion << <mGrid1, 128, 0, CStreams[i] >> > (Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//不同流中的核函数用同一GPU数据时，是否会影响核函数的性能qwt
					GetRecInfo << <mGrid2, 128, 0, CStreams[i] >> > (gpRDevRecData[i], DevPadding[i], DevCounter[i],
						gpDevLength[i], gpDevArea[i], gpDevXpos[i], gpDevYpos[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)*   gRecNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)*   gRecNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(double)*  gRecNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(double)*  gRecNum, cudaMemcpyDeviceToHost, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaStreamSynchronize(CStreams[i]);
				}
				for (int j = 0, int i = 0; i < CUDAStreams; i++)
				{
					for (int k = 0; k < Devpar.PictureNum; k++)
					{
						int hostindex = 0;
						int headpos = myInfo.size();
						CircleInfo headInfo;
						headInfo.index = OutPutInitialIndex + img_index + i;//对应文件索引
						headInfo.xpos = 99999;
						headInfo.ypos = 99999;//xpos 和 ypos作为头标志位
						headInfo.area = 0;	  //area为0也作为特征标志位
						myInfo.push_back(headInfo);
						while (gpHostXpos[i][j] < (k + 1)*Devpar.ImgHeight&&j < gRecNum)
						{
							if (0 < gpHostXpos[i][j])
							{
								hostindex++;
								CircleInfo temp;
								temp.index = hostindex;
								temp.length = gpHostLength[i][j];
								temp.area = gpHostArea[i][j];
								temp.xpos = gpHostXpos[i][j];
								temp.ypos = gpHostYpos[i][j];
								myInfo.push_back(temp);
							}
							j++;
						}
						myInfo[headpos].length = hostindex;//长度置位
					}
				}
				img_index += HardwarePar.CUDAStreamNum*Devpar.PictureNum;
			}
			//写磁盘
			if (myInfo.size() > 0)
			{
				FILE* fp;
				sprintf_s(DataFilename, "%s\\%d.bin", path, OutPutInitialIndex); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
				fp = fopen(DataFilename, "wb");
				fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
				fclose(fp);
			}
			//更新包围盒
			if (DevUpdateRec[HardwarePar.DeviceID] == true)
			{
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], gRecNum * sizeof(RecData), cudaMemcpyHostToDevice);
				}
				DevUpdateRec[HardwarePar.DeviceID] = false;
			}
		}

		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaFree(DevPicColor[i]);
			cudaFree(DevPicGray[i]);
			cudaFree(DevPadding[i]);
			cudaFree(Dev2Val[i]);
			cudaFree(DevCounter[i]);
			cudaFreeHost(gpHostLength[i]);
			cudaFreeHost(gpHostArea[i]);
			cudaFreeHost(gpHostXpos[i]);
			cudaFreeHost(gpHostYpos[i]);
			//设备端内存
			cudaFree(gpDevLength[i]);
			cudaFree(gpDevArea[i]);
			cudaFree(gpDevXpos[i]);
			cudaFree(gpDevYpos[i]);
			cudaFree(gpRDevRecData[i]);
			cudaStreamDestroy(CStreams[i]);
		}

	}
};
int RecR::mRecindex = 0;

/*----------------------------------矩形盒更新类------------------------------------------*/
class RecUpData : public Runnable
{

public:
	Parameter Devpar;//变量传参	
	~RecUpData()
	{
	}
	void Run()
	{

		char strFilename[250];
		//初始化图像信息参数
		Devpar.ImgHeight = gStructVarible.ImgHeight;
		Devpar.ImgWidth = gStructVarible.ImgWidth;
		Devpar.Threshold = gStructVarible.Threshold;
		Devpar.LengthMin = gStructVarible.LengthMin;
		Devpar.LengthMax = gStructVarible.LengthMax;
		Devpar.AreaMin = gStructVarible.AreaMin;
		Devpar.AreaMax = gStructVarible.AreaMax;
		Devpar.PictureNum = gStructVarible.PictureNum;
		Devpar.RecPadding = gStructVarible.RecPadding;
		//方位数组申明
		const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
		//图像空间分配
		unsigned char *ImgHostdata = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum]; //qwt这里程序有BUG
		unsigned char *m_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//二值化图
		unsigned char *n_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//膨胀图
		unsigned char *c_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//轮廓图	
		unsigned char *temp_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//临时变量图

		RecupdataInitialSuccessFlag = true;

		while (ExtractPointSuccess == false)//这里应该加监听（使得提取包围盒可以结束而不是死循环）**************************qwt10.26
		{
			if (HostUpdateRec)//如果缓冲区里面的数据更新了一次 ，则提取包围盒
			{
				vector<RecData>myTempRec;
				memcpy(ImgHostdata, gRecupImgData, sizeof(unsigned char)*Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum);//这个是在内存区域拷贝图像
																															  //二值化
				for (int i = 0; i < Devpar.ImgHeight*Devpar.PictureNum; i++)
				{
					for (int j = 0; j < Devpar.ImgWidth; j++)
					{
						m_ptr[j + i * Devpar.ImgWidth] = ImgHostdata[j + i * Devpar.ImgWidth] > Devpar.Threshold ? 255 : 0;
						c_ptr[j + i * Devpar.ImgWidth] = m_ptr[j + i * Devpar.ImgWidth];
						n_ptr[j + i * Devpar.ImgWidth] = m_ptr[j + i * Devpar.ImgWidth];
						temp_ptr[j + i * Devpar.ImgWidth] = m_ptr[j + i * Devpar.ImgWidth];
					}
				}
				//膨胀
				for (int i = 1; i < Devpar.ImgHeight*Devpar.PictureNum - 1; i++)
					for (int j = 1; j < Devpar.ImgWidth - 1; j++)
					{
						if (m_ptr[j + i * Devpar.ImgWidth] == 0)
						{
							if (m_ptr[j - 1 + (i - 1)*Devpar.ImgWidth] != 0 || m_ptr[j + (i - 1)*Devpar.ImgWidth] != 0 || m_ptr[j + 1 + (i - 1)*Devpar.ImgWidth] != 0 ||
								m_ptr[j - 1 + i * Devpar.ImgWidth] != 0 || m_ptr[j + 1 + i * Devpar.ImgWidth] != 0 ||
								m_ptr[j - 1 + (i + 1)*Devpar.ImgWidth] != 0 || m_ptr[j + (i + 1)*Devpar.ImgWidth] != 0 || m_ptr[j + 1 + (i + 1)*Devpar.ImgWidth] != 0)
							{
								n_ptr[j + i * Devpar.ImgWidth] = 255;
								c_ptr[j + i * Devpar.ImgWidth] = 255;
								temp_ptr[j + i * Devpar.ImgWidth] = 255;
							}
						}
					}
				//腐蚀  c_ptr是轮廓
				for (int i = 1; i < Devpar.ImgHeight*Devpar.PictureNum - 1; i++)
					for (int j = 1; j < Devpar.ImgWidth - 1; j++)
					{
						if (n_ptr[j + i * Devpar.ImgWidth] != 0)
						{
							if (n_ptr[j + (i - 1)*Devpar.ImgWidth] != 0 && n_ptr[j - 1 + i * Devpar.ImgWidth] != 0 &&
								n_ptr[j + 1 + i * Devpar.ImgWidth] != 0 && n_ptr[j + (i + 1)*Devpar.ImgWidth] != 0)
							{
								c_ptr[j + i * Devpar.ImgWidth] = 0;
								temp_ptr[j + i * Devpar.ImgWidth] = 0;
							}
						}
					}
				//方位盒
				short xmax;
				short xmin;
				short ymax;
				short ymin;
				// 边缘跟踪  
				int i, j, counts = 0, curr_d = 0;//counts用于循环计数   curr_d是方向数组的索引ID
				short cLength;
				//提取方位盒子
				for (i = 1; i < Devpar.ImgHeight*Devpar.PictureNum - 1; i++)
					for (j = 1; j < Devpar.ImgWidth - 1; j++)
					{
						// 起始点及当前点  
						cv::Point b_pt = cv::Point(i, j);
						cv::Point c_pt = cv::Point(i, j);
						// 如果当前点为前景点  
						if (255 == c_ptr[j + i * Devpar.ImgWidth])
						{
							cLength = 1;
							xmin = xmax = i;
							ymin = ymax = j;
							/*	bool first_t = false;*/
							bool tra_flag = false;//设置标志位
							c_ptr[j + i * Devpar.ImgWidth] = 0;// 用过的点直接给设置为0  
							while (!tra_flag)// 进行跟踪 
							{
								// 循环八次  
								for (counts = 0; counts < 8; counts++)
								{
									// 防止索引出界  
									if (curr_d >= 8)
									{
										curr_d -= 8;
									}
									if (curr_d < 0)
									{
										curr_d += 8;
									}
									// 跟踪的过程，应该是个连续的过程，需要不停的更新搜索的root点  
									c_pt = cv::Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);
									// 边界判断  
									if ((c_pt.x > 0) && (c_pt.x < Devpar.ImgHeight*Devpar.PictureNum - 1) &&
										(c_pt.y > 0) && (c_pt.y < Devpar.ImgWidth - 1))
									{
										// 如果存在边缘  
										if (255 == c_ptr[c_pt.x*Devpar.ImgWidth + c_pt.y])
										{
											//更新包围盒
											xmax = xmax > c_pt.x ? xmax : c_pt.x;
											ymax = ymax > c_pt.y ? ymax : c_pt.y;
											xmin = xmin < c_pt.x ? xmin : c_pt.x;
											ymin = ymin < c_pt.y ? ymin : c_pt.y;
											curr_d -= 2;   //更新当前方向  
											c_ptr[c_pt.x*Devpar.ImgWidth + c_pt.y] = 0;
											// 更新b_pt:跟踪的root点  
											b_pt.x = c_pt.x;
											b_pt.y = c_pt.y;
											cLength++;
											break;   // 跳出for循环  
										}
									}
									curr_d++;
								}   // end for  
									// 跟踪的终止条件：如果8邻域都不存在边缘  
								if (8 == counts)
								{
									// 清零  
									curr_d = 0;
									tra_flag = true;
									//筛选方位盒
									if (cLength < Devpar.LengthMax && (cLength > Devpar.LengthMin))
									{
										RecData tempRecData;
										int tempcount = 0;
										if (0.7<double(xmax - xmin) / double(ymax - ymin) < 1.5)//高/宽
										{
											for (int k = xmin; k <= xmax; k++)//判断Height方向
											{
												tempcount += temp_ptr[(ymax + ymin) / 2 + k*Devpar.ImgWidth] > 0 ? 1 : 0;
											}
											for (int k = ymin; k <= ymax; k++)//判断width方向
											{
												tempcount += temp_ptr[k + (xmax + xmin) / 2 * Devpar.ImgWidth] > 0 ? 1 : 0;
											}
											if (tempcount <= 4)
											{
												if (xmin - Devpar.RecPadding < 0)
													tempRecData.RecXmin = 0;
												else
													tempRecData.RecXmin = xmin - Devpar.RecPadding;
												if (ymin - Devpar.RecPadding < 0)
													tempRecData.RecYmin = 0;
												else
													tempRecData.RecYmin = ymin - Devpar.RecPadding;
												if (xmax + Devpar.RecPadding > Devpar.ImgHeight*Devpar.PictureNum - 1)
													tempRecData.RecXmax = Devpar.ImgHeight*Devpar.PictureNum - 1;
												else
													tempRecData.RecXmax = xmax + Devpar.RecPadding;
												if (ymax + Devpar.RecPadding > Devpar.ImgWidth)
													tempRecData.RecYmax = Devpar.ImgWidth - 1;
												else
													tempRecData.RecYmax = ymax + Devpar.RecPadding;
												myTempRec.push_back(tempRecData);
											}
										}
									}
									break;
								}
							}  // end if  
						}  // end while  
					}
				//规整方位盒数量，利用后续线程配置
				gSingleImgRecNum = myTempRec.size() / Devpar.PictureNum;//单张图方位盒数量
				int rRecNum = (myTempRec.size() + 127) / 128 * 128;
				myTempRec.resize(gRecNum, RecData{ 0,0,0,0 });
				if (gRecNum != 0)
				{
					memcpy(&gHostRecData[0], &myTempRec[0], sizeof(RecData)*gRecNum);
					for (int m = 0; m < HardwareParam.DeviceCount; m++)
					{
						DevUpdateRec[m] = true;
					}
				}
				HostUpdateRec = false;
			}
		}
		//释放内存
		delete[]ImgHostdata;
		delete[]m_ptr;
		delete[]n_ptr;
		delete[]c_ptr;
		delete[]temp_ptr;
	}
};

/*----------------------------------实现彩图压缩功能的类----------------------------------*/
class TC : public Runnable
{
public:
	HardwareInfo param;									//硬件参数
	unsigned char* my_in;								//显存中的原始位图数据
	needmemory memory;									//压缩程序所需显存
	needdata staticdata;
	static int mTCindex;
	unsigned char* total_malloc;						//每一包二进制文件占用内存
	int pix_index;
public:
	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	/*************************************************************************************************
	Function:       void Initialize()
	Description:    用来初始化数据结构和分配显存空间的成员函数
	Calls:          cudaMalloc()、nppiDCTInitAlloc()、cudaMemcpyAsync()、cudaMallocPitch()、
	nppiEncodeHuffmanSpecInitAlloc_JPEG()，它们都是cuda库中的函数

	Input:          无
	Output:         无
	***************************************************************************************************/
	void Initialize()
	{
		//cudaMalloc((void**)&(this->my_in), imgHeight * imgWidth * sizeof(unsigned char) * 3);	//为my_in分配显存空间
		cudaMalloc((void**)&(this->my_in), compress_old_Height * compress_old_Width * sizeof(unsigned char) * 3);
		nppiDCTInitAlloc(&(this->memory).pDCTState);											//为memory.pDCTState分配显存空间
		cudaMalloc(&(this->staticdata).pdQuantizationTables, 64 * 4);							//staticdata.pdQuantizationTables分配显存空间

		float nScaleFactor;
		nScaleFactor = 1.0f;
		int nMCUBlocksH = 0;
		int nMCUBlocksV = 0;
		quantityassgnment();

		for (int i = 0; i < oFrameHeader.nComponents; ++i)
		{
			nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
			nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
		}


		Npp8u aZigzag[] = {
			0,  1,  5,  6, 14, 15, 27, 28,
			2,  4,  7, 13, 16, 26, 29, 42,
			3,  8, 12, 17, 25, 30, 41, 43,
			9, 11, 18, 24, 31, 40, 44, 53,
			10, 19, 23, 32, 39, 45, 52, 54,
			20, 22, 33, 38, 46, 51, 55, 60,
			21, 34, 37, 47, 50, 56, 59, 61,
			35, 36, 48, 49, 57, 58, 62, 63
		};

		for (int i = 0; i < 4; ++i)
		{
			Npp8u temp[64];

			for (int k = 0; k < 32; ++k)
			{
				temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k + 0]];
				temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
			}

			cudaMemcpyAsync((unsigned char *)(this->staticdata).pdQuantizationTables + i * 64, temp, 64, cudaMemcpyHostToDevice);

		}

		float frameWidth = floor((float)oFrameHeader.nWidth * (float)nScaleFactor);
		float frameHeight = floor((float)oFrameHeader.nHeight * (float)nScaleFactor);

		(this->staticdata).oDstImageSize.width = (int)max(1.0f, frameWidth);
		(this->staticdata).oDstImageSize.height = (int)max(1.0f, frameHeight);

		size_t newPitch[3];
		NppiSize oBlocks;


		for (int i = 0; i < oFrameHeader.nComponents; ++i)								//根据图像大小计算一些参数，之后在DCT变换和Huffman编码中要用到
		{
			//NppiSize oBlocks;
			NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] & 0x0f, oFrameHeader.aSamplingFactors[i] >> 4 };

			oBlocks.width = (int)ceil(((this->staticdata).oDstImageSize.width + 7) / 8 *
				static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
			oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

			oBlocks.height = (int)ceil(((this->staticdata).oDstImageSize.height + 7) / 8 *
				static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
			oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

			(this->staticdata).aDstSize[i].width = oBlocks.width * 8;
			(this->staticdata).aDstSize[i].height = oBlocks.height * 8;
		}


		// Scale to target image size
		// Assume we only deal with 420 images.
		int aSampleFactor[3] = { 1, 2, 2 };

		(this->memory).nScanSize = (this->staticdata).oDstImageSize.width * (this->staticdata).oDstImageSize.height * 2;
		(this->memory).nScanSize = (this->memory).nScanSize > (4 << 20) ? (this->memory).nScanSize : (4 << 20);
		cudaMalloc(&(this->memory).pDScan, (this->memory).nScanSize);														//为memory.pDScan分配显存空间
		nppiEncodeHuffmanGetSize((this->staticdata).aDstSize[0], 3, &(this->memory).nTempSize);
		cudaMalloc(&(this->memory).pDJpegEncoderTemp, (this->memory).nTempSize);											//为memory.pDJpegEncoderTemp分配显存空间


		for (int j = 0; j < 3; j++) {
			size_t nPitch1;
			cudaMallocPitch(&(this->memory).pDCT[j], &nPitch1, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height);		//为memory.pDCT分配内存空间
			(this->memory).DCTStep[j] = static_cast<Npp32s>(nPitch1);
			cudaMallocPitch(&(this->memory).pDImage[j], &nPitch1, (this->staticdata).aDstSize[j].width, (this->staticdata).aDstSize[j].height);		//为memory.pDImage分配显存空间
			(this->memory).DImageStep[j] = static_cast<Npp32s>(nPitch1);
			dataduiqi[j] = nPitch1;

		}
		for (int i = 0; i < 3; ++i)				//初始化显存中的staticdata.apDHuffmanDCTable 和 staticdata.apDHuffmanACTable
		{
			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &(this->staticdata).apDHuffmanDCTable[i]);
			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &(this->staticdata).apDHuffmanACTable[i]);
		}

		for (int iComponent = 0; iComponent < 2; ++iComponent)
		{
			(this->memory).hpCodesDC[iComponent] = pHuffmanDCTables[iComponent].aCodes;
			(this->memory).hpCodesAC[iComponent] = pHuffmanACTables[iComponent].aCodes;
			(this->memory).hpTableDC[iComponent] = pHuffmanDCTables[iComponent].aTable;
			(this->memory).hpTableAC[iComponent] = pHuffmanACTables[iComponent].aTable;
		}
	}

	/*************************************************************************************************
	Function:       void process()
	Description:    首先调用jpegNPP工程的nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW函数，
	作用是对memory.pDImage中的照片YUV数据进行DCT变换和量化，并将结果保存在memory.pDCT中；

	之后调用jpegNPP工程的nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R函数，
	作用是对经过DCT变换后的图像数据memory.pDCT进行霍夫曼编码，编码后的数据在memory.pDScan中保存，等待写入磁盘。

	Calls:          nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW()、nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R()，它们都是cuda库中的函数

	Input:          无
	Output:         无
	***************************************************************************************************/
	void process()
	{
		for (int i = 0; i < 3; ++i)													//对YCbCr三个通道的图片数据进行DCT变换
		{
			nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW((this->memory).pDImage[i], (this->memory).DImageStep[i],
				(this->memory).pDCT[i], (this->memory).DCTStep[i],
				(this->staticdata).pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
				(this->staticdata).aDstSize[i],
				(this->memory).pDCTState);
		}

		nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R((this->memory).pDCT, (this->memory).DCTStep,			//进行霍夫曼编码操作
			0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
			(this->memory).pDScan, &(this->memory).nScanLength,
			(this->memory).hpCodesDC, (this->memory).hpTableDC, (this->memory).hpCodesAC, (this->memory).hpTableAC,
			(this->staticdata).apDHuffmanDCTable,
			(this->staticdata).apDHuffmanACTable,
			(this->staticdata).aDstSize,
			(this->memory).pDJpegEncoderTemp);
	}

	/*************************************************************************************************
	Function:       void writedisk()
	Description:    完成jpg图片写入磁盘的工作。
	先把JFIF标签、aQuantizationTables标准量化表、oFrameHeader结构头、
	标准霍夫曼编码表、oScanHeader扫描头写入文件作为jpg图片的文件头，
	之后将memory.pDScan编码后的数据输入写入文件，组成完整的.jpg图片。


	Calls:          writeMarker()、writeJFIFTag()、writeQuantizationTable()、writeHuffmanTable()
	这些函数定义在useful.h中

	Input:          无
	Output:         无
	***************************************************************************************************/
	//void writedisk(char* OutputFile)
	void writedisk(int picture_num, Package* a, int bag_index)
	{
		unsigned char *pDstJpeg = new unsigned char[(this->memory).nScanSize];							//为每一张.jpg图片数据开辟缓冲区
		unsigned char *pDstOutput = pDstJpeg;

		oFrameHeader.nWidth = (this->staticdata).oDstImageSize.width;
		oFrameHeader.nHeight = (this->staticdata).oDstImageSize.height;

		writeMarker(0x0D8, pDstOutput);
		writeJFIFTag(pDstOutput);
		writeQuantizationTable(aQuantizationTables[0], pDstOutput);										//写入标准量化表
		writeQuantizationTable(aQuantizationTables[1], pDstOutput);
		writeFrameHeader(oFrameHeader, pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);												//写入霍夫曼编码表
		writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
		writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
		writeScanHeader(oScanHeader, pDstOutput);

		cudaMemcpy(pDstOutput, (this->memory).pDScan, (this->memory).nScanLength, cudaMemcpyDeviceToHost);
		pDstOutput += (this->memory).nScanLength;
		writeMarker(0x0D9, pDstOutput);
		char szOutputFiler[100];
		sprintf_s(szOutputFiler, "%s\\%d.jpg", gStructVarible.ImgSavePath, picture_num);

		memcpy(total_malloc + pix_index, pDstJpeg, static_cast<int>(pDstOutput - pDstJpeg));			//将这一整张.jpg图片数据拷贝到大的内存区total_malloc
		pix_index += static_cast<int>(pDstOutput - pDstJpeg);
		//a->Form_one_head(bag_index / gStructVarible.PictureNum, szOutputFiler, pDstOutput - pDstJpeg);

		a->Form_one_head(bag_index / gStructVarible.PictureNum, picture_num, pDstOutput - pDstJpeg);					//完成一张.jpg图片对应的包头， bag_index / gStructVarible.PictureNum代表这是第几张图

																														//{
																														//Write result to file.
																														//std::ofstream outputFile1(OutputFile, ios::out | ios::binary);
																														//outputFile1.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
																														//}

		delete[] pDstJpeg;
	}


	/*************************************************************************************************
	Function:       void memoryfree()
	Description:    程序结束后，释放之前分配好的显存空间
	Calls:          cudaFree()、nppiEncodeHuffmanSpecFree_JPEG()、nppiDCTFree()
	它们都是cuda库中的函数

	Input:          无
	Output:         无
	***************************************************************************************************/
	void memoryfree()																	//释放之前申请的内存和显存
	{
		cudaFree(this->my_in);
		for (int i = 0; i < 3; ++i)
		{
			cudaFree((this->memory).pDCT[i]);
			cudaFree((this->memory).pDImage[i]);
			nppiEncodeHuffmanSpecFree_JPEG((this->staticdata).apDHuffmanDCTable[i]);
			nppiEncodeHuffmanSpecFree_JPEG((this->staticdata).apDHuffmanACTable[i]);
		}
		nppiDCTFree((this->memory).pDCTState);
		cudaFree((this->memory).pDJpegEncoderTemp);
		cudaFree((this->memory).pDScan);
		cudaFree((this->staticdata).pdQuantizationTables);
	}

	~TC() {}
	/*************************************************************************************************
	Function:       Run()
	Description:    是多线程类T运行的入口函数，整个压缩模块从这里开始运行
	Calls:          依次调用了Initialize()、RGBtoYUV <<<blocks, threads >>>、process()、
	writedisk(szOutputFile)和memoryfree()

	Input:          无
	Output:         无
	***************************************************************************************************/
	void Run()
	{
		char ImgoutputPath[255];
		total_malloc = new unsigned char[100000000];
		pix_index = 0;
		char szOutputFile[100];
		clock_t start, end;
		int img_index;									//图像索引
		int mFlagIndex = 0;
		int OutPutInitialIndex = 0;						//输出的Bin文件初始索引号
		int Bufferoffset = 0;							//缓冲区偏移量
		bool DatafullFlag = false;		//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。

		cudaSetDevice((this->param).GpuId);
		this->Initialize();

		cout << "T GPU ：" << param.GpuId << " initial success!" << endl;

		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			img_index = 0;								//图像计数
			Bufferoffset = 0;
			//获取数据
			while (true)
			{
				gComressReadDataLock.lock();
				mTCindex = mTCindex % (HardwareParam.DeviceCount + 1);
				if (gComressionBufferEmpty[mTCindex] == false && gComressionBufferWorking[mTCindex] == false)
				{
					//将页锁内存标志位置为工作状态--进行绑定
					gComressionBufferWorking[mTCindex] = true;
					OutPutInitialIndex = gComressionBufferStartIndex[mTCindex] * Bufferlength;//获取图像首索引
					mFlagIndex = mTCindex;
					DatafullFlag = true;
					mTCindex++;
					gComressReadDataLock.unlock();
					break;
				}
				mTCindex++;
				gComressReadDataLock.unlock();
				if (ExtractPointSuccess)
					break;
			}
			start = clock();
			sprintf_s(ImgoutputPath, "%s\\%d.bin", gStructVarible.ImgSavePath, OutPutInitialIndex);
			cout << ImgoutputPath << endl;
			//Package data_bag(ImgoutputPath, Bufferlength / gStructVarible.PictureNum);
			Package data_bag(ImgoutputPath);
			data_bag.Package_init(Bufferlength / gStructVarible.PictureNum);

			//压缩pImg图片
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength)
				{
					end = clock();
					gComressReadDataLock.lock();
					gComressionBufferWorking[mFlagIndex] = false;
					gComressReadDataLock.unlock();
					gComressionBufferEmpty[mFlagIndex] = true;
					DatafullFlag = false;


					compress_write_lock.lock();
					data_bag.file.open(data_bag.Fname, ios::out | ios::binary);										//50张.jpg完成后，打开一个二进制文件
																													//data_bag.Form_total_head();																		//完成所有50张图片的包头信息
					data_bag.Form_total_head(compress_imgWidth, compress_imgHeight, gStructVarible.PictureNum, OutPutInitialIndex);
					data_bag.file.write(data_bag.head_cache, data_bag.head_bias);									//写入所有包头
					data_bag.file.write(reinterpret_cast<const char *>(total_malloc), static_cast<int>(pix_index)); //写入所有数据
					data_bag.file.close();
					//data_bag.UnPack(data_bag.Fname);
					compress_write_lock.unlock();
					memset(total_malloc, 0, 100000000);																//缓冲区清空
					pix_index = 0;																					//缓冲区索引归零
					break;
				}
				//sprintf_s(szOutputFile, "%s\\%d.jpg", gStructVarible.ImgSavePath, OutPutInitialIndex + img_index);
				int picture_index = OutPutInitialIndex + img_index;
				Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.PictureNum * 3;
				cudaMemcpy(this->my_in, gHostComressiongBuffer[mFlagIndex] + Bufferoffset, compress_old_Width * compress_old_Height * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
				RGBtoYUV << <blocks, threads >> > (this->my_in, (this->memory).pDImage[0], (this->memory).pDImage[2], (this->memory).pDImage[1], compress_imgHeight, compress_imgWidth, dataduiqi[0], compress_old_Height, compress_old_Width);
				this->process();
				this->writedisk(picture_index, &data_bag, img_index);
				//img_index++;
				img_index = img_index + gStructVarible.PictureNum;                                                //picture_index是一次实验总的图片标号
			}
		}
		delete[] total_malloc;
		this->memoryfree();
	}
};
int TC::mTCindex = 0;

//----------------------------------实现灰度图压缩功能的类------------------------------------------//
//核函数自编版
class T : public Runnable									//由于该类的实现和前者TC类及其相似，所以不再进行注释
{
public:
	HardwareInfo param;										//硬件参数
	gpuneedmemory memory[GRAYCompressStreams];
	static int mTindex;
	needconstdata staticdata;                               //压缩过程中用到的常量数据
	RIM   ImageSize;                                        //记录图像大小，和每个图像分量对齐后的对齐宽度
	size_t Org_Pitch;                                       //记录原始图像数据对齐后的宽度
	int  h_MCUtotal;                                        //单个图像分量8*8像素块总量
	cudaStream_t stream[GRAYCompressStreams];               //申明CUDA流
	int   stridef;
	cpuneedmemory cpumemory[GRAYCompressStreams];           //存放压缩图像过程中CPU上的原始图像位图数据和最终编码图像数据
	unsigned char* total_malloc;                            //打包数据在内存中的缓存空间
	int pix_index;
public:
	void mydelay(double sec)                               //延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	//*************************************************************************************************
	//*Function:       void Initialize()
	//*Description:    用来初始化数据结构和分配显存空间的成员函数
	//*Calls:          cudaMalloc()、cudaMemcpyAsync()、cudaMallocPitch()它们都是cuda库中的函数
	//*Input:          ImageSize 记录图像数据大小，用于分配显存与内存
	//*Output:         无
	//***************************************************************************************************
	void Initialize() {
		size_t nPitch;
		this->stridef = ALIGN(compress_old_Width, 4);
		(this->ImageSize).width = ALIGN(compress_old_Width, 8);
		(this->ImageSize).height = ALIGN(compress_old_Height, 8);
		int h_MCUtotal = (this->ImageSize).height*(this->ImageSize).width / 64;
		int ARRAY_SIZE = ALIGN(h_MCUtotal + 1025, 1024);
		int ARRAY_SIZE1 = ALIGN(h_MCUtotal / 1024 + 1025, 1024);

		//为最后编码的图像数据确定内存大小
		(this->staticdata).nScanSize = (this->ImageSize).width * (this->ImageSize).height * 2;
		(this->staticdata).nScanSize = (this->staticdata).nScanSize > (10 << 20) ? (this->staticdata).nScanSize : (10 << 20);

		for (int i = 0; i < GRAYCompressStreams; i++) {
			//为每一个流分配显存与内存
			cudaMallocPitch((void **)&(this->memory[i].d_bsrc), &(this->ImageSize.StrideF), (this->ImageSize).width * sizeof(BYTE), (this->ImageSize).height);      //为my_in分配显存空间
			cudaMallocPitch((void **)&(this->memory[i].d_ydst), &nPitch, (this->ImageSize).width * (this->ImageSize).height * sizeof(BSI16), 1);
			cudaMallocPitch((void **)&(this->memory[i].d_JPEGdata), &nPitch, (this->ImageSize).width * sizeof(BYTE)*(this->ImageSize).height, 1);
			cudaMalloc((void **)&(this->memory[i].last_JPEGdata), (10 << 20));
			cudaMalloc((void **)&(this->memory[i].prefix_num), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].last_prefix_num), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].dc_component), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].d_blocksum), 768 * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].d_datalen), sizeof(int));
			//创建CUDA流
			cudaStreamCreate(&(this->stream[i]));
			//分配CPU内存
			//cudaHostAlloc((BYTE**)&(this->cpumemory[i]).pDstJpeg, (this->staticdata).nScanSize, cudaHostAllocDefault);    //最终编码数据
			(this->cpumemory[i]).pDstJpeg = new unsigned char[(this->staticdata).nScanSize];
			this->cpumemory[i].pDstOutput = this->cpumemory[i].pDstJpeg;
		}

		//-------------------------为灰度图像压缩配置常量数据--------------------
		cudaMalloc(&(this->staticdata).DEV_STD_QUANT_TAB_LUMIN, 64 * sizeof(float));
		cudaMalloc(&(this->staticdata).DEV_ZIGZAG, 64 * sizeof(int));
		{
			//--------------------配置亮度量化表--------------------------------
			float temp[64];
			for (int i = 0; i<64; i++) {
				temp[i] = 1.0f / (float)STD_QUANT_TAB_LUMIN[i] * C_norm * C_norm;
			}
			cudaMemcpyAsync((this->staticdata).DEV_STD_QUANT_TAB_LUMIN, temp, 64 * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMemcpyAsync((this->staticdata).DEV_ZIGZAG, aZIGZAG, 64 * sizeof(float), cudaMemcpyHostToDevice);
		{
			//----------------初始化huffman表
			GPUjpeg_huffman_encoder_value_init_kernel << <32, 256 >> >();  // 8192 threads total
																		   // 创建GPU版本的Huffman表 ( CC >= 2.0)
			uint32_t gpujpeg_huffman_cpu_lut[(256 + 1) * 4];
			memset(gpujpeg_huffman_cpu_lut, 0, (256 + 1) * 4 * sizeof(uint32_t));
			Newhuffman_table_init(gpujpeg_huffman_cpu_lut + 257 * 0, STD_HUFTAB_LUMIN_AC, true);
			Newhuffman_table_init(gpujpeg_huffman_cpu_lut + 257 * 1, STD_HUFTAB_LUMIN_DC, false);
			Newhuffman_table_init(gpujpeg_huffman_cpu_lut + 257 * 2, STD_HUFTAB_CHROM_AC, true);
			Newhuffman_table_init(gpujpeg_huffman_cpu_lut + 257 * 3, STD_HUFTAB_CHROM_DC, false);
			cudaMemcpyToSymbol(gpujpeg_huffman_gpu_tab, gpujpeg_huffman_cpu_lut,
				(256 + 1) * 4 * sizeof(*gpujpeg_huffman_gpu_tab), 0,
				cudaMemcpyHostToDevice
			);
		}
	}
	//**************************************************************************************************
	//**Function:       void process()
	//**Description:    用来压缩图像的函数
	//**Input:          Size 记录图像数据大小，用于分配显存与内存
	//**Output:         无
	//***************************************************************************************************
	void process() {
		const int ARRAY_SIZE = ImageSize.width * ImageSize.height;
		const int  h_MCUtotal = ARRAY_SIZE / 64;                               //图像数据总的8*8MCU单元

		const int Code_blocks = (h_MCUtotal + CODE_THREADS - 1) / CODE_THREADS;
		int Blocksums;
		int prexsum_blocks = 1;
		int prexsum_threads = (h_MCUtotal - 1) / CODE_THREADS;

		//prefix_sum前缀求和线程分配
		int preSum_Blocks = (h_MCUtotal + 1023) / 1024;
		//DCT线程分配
		dim3 DCT_blocks((ImageSize.width + 63) / DCT_BLOCK_WIDTH, ImageSize.height / DCT_BLOCK_HEIGHT);
		dim3 DCT_threads(8, 32 / 8, 2);

		dim3 Encode_thread(THREAD_WARP, 4);
		dim3 Encode_Blocks(gpujpeg_huffman_encoder_grid_size((h_MCUtotal + 3) / 4));

		for (int i = 0; i < GRAYCompressStreams; i++) {
			CUDA_DCT8_kernel << <DCT_blocks, DCT_threads, 0, this->stream[i] >> >(this->memory[i].d_ydst,
				this->memory[i].d_bsrc, ImageSize, this->staticdata.DEV_ZIGZAG,
				this->staticdata.DEV_STD_QUANT_TAB_LUMIN);
		}
		for (int i = 0; i < GRAYCompressStreams; i++) {
			Data_codelength_kernel << <Encode_Blocks, Encode_thread, 0, this->stream[i] >> > (this->memory[i].d_ydst,
				h_MCUtotal, this->memory[i].d_JPEGdata, this->memory[i].prefix_num, 1, 0);

			//计算每个mcu比特流的具体位置，前缀求和算法
			work_efficient_PrefixSum_kernel << <preSum_Blocks, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].prefix_num, this->memory[i].dc_component);
			if (h_MCUtotal <= PRESUM_THREADS * 512) {
				work_efficient_BlockUp_kernel << <1, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component);
				work_efficient_Adds_kernel << <(h_MCUtotal + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].prefix_num);
			}
			else {
				work_efficient_PrefixSum_kernel << < ((h_MCUtotal - 1) / 512 + 1023) / 1024, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].d_blocksum);
				work_efficient_BlockUp_kernel << <1, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].d_blocksum);
				work_efficient_Adds_kernel << <((h_MCUtotal + 511) / 512 + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].d_blocksum, this->memory[i].dc_component);
				work_efficient_Adds_kernel << <(h_MCUtotal + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].prefix_num);
			}

			//对图像数据进行编码处理
			data_shift_kernel << <Code_blocks, CODE_THREADS, 0, this->stream[i] >> >(this->memory[i].d_JPEGdata,
				this->memory[i].prefix_num, h_MCUtotal, this->memory[i].d_datalen,
				this->memory[i].dc_component, this->memory[i].last_prefix_num);
			//计算每个MCU BYTE流的具体位置，前缀求和算法
			work_efficient_PrefixSum_kernel << <preSum_Blocks, PRESUM_THREADS, 0, this->stream[i] >> > (this->memory[i].last_prefix_num, this->memory[i].dc_component);
			if (h_MCUtotal <= PRESUM_THREADS * 512) {
				work_efficient_BlockUp_kernel << <1, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component);
				work_efficient_Adds_kernel << <(h_MCUtotal + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].last_prefix_num);
			}
			else {
				work_efficient_PrefixSum_kernel << < ((h_MCUtotal - 1) / 512 + 1023) / 1024, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].d_blocksum);
				work_efficient_BlockUp_kernel << <1, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].d_blocksum);
				work_efficient_Adds_kernel << <((h_MCUtotal + 511) / 512 + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].d_blocksum, this->memory[i].dc_component);
				work_efficient_Adds_kernel << <(h_MCUtotal + 511) / 512, PRESUM_THREADS, 0, this->stream[i] >> >(this->memory[i].dc_component, this->memory[i].last_prefix_num);
			}
			cudaMemsetAsync(this->memory[i].last_JPEGdata, 0, (10 << 20), this->stream[i]);

			Data_encodelater1_kernel << <Code_blocks, CODE_THREADS, 0, this->stream[i] >> >(this->memory[i].last_prefix_num,
				this->memory[i].d_JPEGdata, this->memory[i].last_JPEGdata, h_MCUtotal,
				this->memory[i].d_datalen);

			//得到图像数据编码长度
			cudaMemcpyAsync(&this->cpumemory[i].dst_JPEGdatalength, (this->memory[i]).d_datalen, sizeof(int), cudaMemcpyDeviceToHost, this->stream[i]);
		}
	}

	void writedisk(int picture_num, Package* a, int bag_index)
	{
		for (int i = 0; i < GRAYCompressStreams; i++) {
			this->cpumemory[i].pDstOutput = this->cpumemory[i].pDstJpegDataStart;
			//编码后图像数据传输
			cudaMemcpyAsync(this->cpumemory[i].pDstOutput, this->memory[i].last_JPEGdata,
				this->cpumemory[i].dst_JPEGdatalength,
				cudaMemcpyDeviceToHost, this->stream[i]);
			//-------------等待Stream流执行完成
			cudaStreamSynchronize(this->stream[i]);

			this->cpumemory[i].pDstOutput += this->cpumemory[i].dst_JPEGdatalength;
			writeMarker(0x0D9, this->cpumemory[i].pDstOutput);

			//char szOutputFiler[100];
			//sprintf_s(szOutputFiler, "%s\\%d.jpg", gStructVarible.ImgSavePath, picture_num);
			memcpy(total_malloc + pix_index, this->cpumemory[i].pDstJpeg, static_cast<int>(this->cpumemory[i].pDstOutput - this->cpumemory[i].pDstJpeg));
			pix_index += static_cast<int>(this->cpumemory[i].pDstOutput - this->cpumemory[i].pDstJpeg);
			a->Form_one_head(bag_index / gStructVarible.PictureNum, picture_num, this->cpumemory[i].pDstOutput - this->cpumemory[i].pDstJpeg);
			picture_num = picture_num + gStructVarible.PictureNum;
			bag_index = bag_index + gStructVarible.PictureNum;
		}
	}
	void WriteJpgheader() {
		for (int i = 0; i < GRAYCompressStreams; i++) {
			writeMarker(0x0D8, this->cpumemory[i].pDstOutput);
			writeMarker(0x0DB, this->cpumemory[i].pDstOutput);
			writeWords(67, this->cpumemory[i].pDstOutput);
			writeChar(0, this->cpumemory[i].pDstOutput);
			for (int j = 0; j < 64; j++) {
				writeChar(STD_QUANT_TAB_LUMIN[ZIGZAG[j]], this->cpumemory[i].pDstOutput);
			}
			writeMarker(0x0DB, this->cpumemory[i].pDstOutput);
			writeWords(67, this->cpumemory[i].pDstOutput);
			writeChar(1, this->cpumemory[i].pDstOutput);
			for (int j = 0; j < 64; j++) {
				writeChar(STD_QUANT_TAB_CHROM[ZIGZAG[j]], this->cpumemory[i].pDstOutput);
			}
			writeMarker(0x0C0, this->cpumemory[i].pDstOutput);
			unsigned short len = 2 + 1 + 2 + 2 + 1 + 3 * 3;   //3是颜色分量数
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(8, this->cpumemory[i].pDstOutput);
			writeWords(compress_old_Height, this->cpumemory[i].pDstOutput);
			writeWords(compress_old_Width, this->cpumemory[i].pDstOutput);
			writeChar(3, this->cpumemory[i].pDstOutput);
			writeChar(1, this->cpumemory[i].pDstOutput);
			writeChar((1 << 0) | (1 << 4), this->cpumemory[i].pDstOutput);
			writeChar(0, this->cpumemory[i].pDstOutput);
			writeChar(2, this->cpumemory[i].pDstOutput);
			writeChar((1 << 0) | (1 << 4), this->cpumemory[i].pDstOutput);
			writeChar(1, this->cpumemory[i].pDstOutput);
			writeChar(3, this->cpumemory[i].pDstOutput);
			writeChar((1 << 0) | (1 << 4), this->cpumemory[i].pDstOutput);
			writeChar(1, this->cpumemory[i].pDstOutput);

			//*********************************************************************************************
			// output DHT AC   0xC4     霍夫曼(Huffman)表 
			writeMarker(0x0C4, this->cpumemory[i].pDstOutput);
			len = 2 + 1 + 16 + 162;
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(0 + 0x10, this->cpumemory[i].pDstOutput);
			memcpy(this->cpumemory[i].pDstOutput, STD_HUFTAB_LUMIN_AC, len - 3);
			this->cpumemory[i].pDstOutput += len - 3;

			writeMarker(0x0C4, this->cpumemory[i].pDstOutput);
			len = 2 + 1 + 16 + 162;
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(1 + 0x10, this->cpumemory[i].pDstOutput);
			memcpy(this->cpumemory[i].pDstOutput, STD_HUFTAB_CHROM_AC, len - 3);
			this->cpumemory[i].pDstOutput += len - 3;

			// output DHT DC 0xC4    霍夫曼(Huffman)表
			writeMarker(0x0C4, this->cpumemory[i].pDstOutput);
			len = 2 + 1 + 16 + 12;
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(0 + 0x00, this->cpumemory[i].pDstOutput);
			memcpy(this->cpumemory[i].pDstOutput, STD_HUFTAB_LUMIN_DC, len - 3);
			this->cpumemory[i].pDstOutput += len - 3;

			writeMarker(0x0C4, this->cpumemory[i].pDstOutput);
			len = 2 + 1 + 16 + 12;
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(1 + 0x00, this->cpumemory[i].pDstOutput);
			memcpy(this->cpumemory[i].pDstOutput, STD_HUFTAB_CHROM_DC, len - 3);
			this->cpumemory[i].pDstOutput += len - 3;

			// output SOS  0xDA　 扫描线开始
			len = 2 + 1 + 2 * 3 + 3;
			writeMarker(0x0DA, this->cpumemory[i].pDstOutput);
			writeWords(len, this->cpumemory[i].pDstOutput);
			writeChar(3, this->cpumemory[i].pDstOutput);

			writeChar(1, this->cpumemory[i].pDstOutput);
			writeChar((0 << 0) | (0 << 4), this->cpumemory[i].pDstOutput);
			writeChar(2, this->cpumemory[i].pDstOutput);
			writeChar((1 << 0) | (1 << 4), this->cpumemory[i].pDstOutput);
			writeChar(3, this->cpumemory[i].pDstOutput);
			writeChar((1 << 0) | (1 << 4), this->cpumemory[i].pDstOutput);

			writeChar(0x00, this->cpumemory[i].pDstOutput);
			writeChar(0x3f, this->cpumemory[i].pDstOutput);
			writeChar(0x00, this->cpumemory[i].pDstOutput);
			this->cpumemory[i].pDstJpegDataStart = this->cpumemory[i].pDstOutput;
		}

	}
	//释放申请的显存与内存
	void memoryfree()
	{
		for (int i = 0; i < GRAYCompressStreams; i++) {
			//释放显存
			cudaFree(this->memory[i].d_bsrc);
			cudaFree(this->memory[i].d_ydst);
			cudaFree(this->memory[i].d_JPEGdata);
			cudaFree(this->memory[i].last_JPEGdata);
			cudaFree(this->memory[i].prefix_num);
			cudaFree(this->memory[i].last_prefix_num);
			cudaFree(this->memory[i].dc_component);
			cudaFree(this->memory[i].d_blocksum);
			cudaFree(this->memory[i].d_datalen);
			//cudaFree(this->cpumemory[i].pDstJpeg);
			delete[] this->cpumemory[i].pDstJpeg;
		}
		cudaFree(this->staticdata.DEV_STD_QUANT_TAB_LUMIN);
		cudaFree(this->staticdata.DEV_ZIGZAG);

	}
	~T() {}
	void Run() {
		char ImgoutputPath[255];
		total_malloc = new unsigned char[100000000];
		pix_index = 0;
		clock_t start, end, end2;
		int img_index;//图像索引
		int cudaStreams_imgindex = 0; //每个流的图像索引
		int mFlagIndex = 0;
		int OutPutInitialIndex = 0; //输出的Bin文件初始索引号
		int Bufferoffset = 0;       //缓冲区偏移量
		bool DatafullFlag = false;//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。
								  //测试读入图片是否成功-------------------------------------------------------------------------------------
		cv::Mat img1(5120, 5120, CV_8UC1);
		cudaSetDevice((this->param).GpuId);
		this->Initialize();
		cout << "T GPU ：" << param.GpuId << " initial success!" << endl;
		WriteJpgheader();
		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			img_index = 0;//图像计数
			Bufferoffset = 0;
			//绑定数据
			while (true)//这里需要改，锁不能用和提点一样
			{
				gComressReadDataLock.lock();
				mTindex = mTindex % (HardwareParam.DeviceCount + 1);
				if (gComressionBufferEmpty[mTindex] == false && gComressionBufferWorking[mTindex] == false)
				{
					//将页锁内存标志位置为工作状态--进行绑定
					gComressionBufferWorking[mTindex] = true;
					OutPutInitialIndex = gComressionBufferStartIndex[mTindex] * Bufferlength;//获取图像首索引
					mFlagIndex = mTindex;
					DatafullFlag = true;
					mTindex++;
					gComressReadDataLock.unlock();
					break;
				}
				mTindex++;
				gComressReadDataLock.unlock();
				if (ExtractPointSuccess)
					break;
			}
			start = clock();
			sprintf_s(ImgoutputPath, "%s\\%d.bin", gStructVarible.ImgSavePath, OutPutInitialIndex);

			Package data_bag(ImgoutputPath);
			data_bag.Package_init(Bufferlength / gStructVarible.PictureNum);

			//压缩pImg图片
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength)
				{
					end = clock();
					gComressReadDataLock.lock();
					gComressionBufferWorking[mFlagIndex] = false;
					gComressReadDataLock.unlock();
					gComressionBufferEmpty[mFlagIndex] = true;
					DatafullFlag = false;

					//写磁盘
					compress_write_lock.lock();
					data_bag.file.open(data_bag.Fname, ios::out | ios::binary);
					//data_bag.Form_total_head();
					data_bag.Form_total_head(compress_imgWidth, compress_imgHeight, gStructVarible.PictureNum, OutPutInitialIndex);
					//cout << OutPutInitialIndex << endl;
					data_bag.file.write(data_bag.head_cache, data_bag.head_bias);
					data_bag.file.write(reinterpret_cast<const char *>(total_malloc), static_cast<int>(pix_index));
					data_bag.file.close();
					//data_bag.UnPack(data_bag.Fname);
					compress_write_lock.unlock();
					memset(total_malloc, 0, 100000000);
					pix_index = 0;
					end2 = clock();
					//cout << "T GPU ：" << param.GpuId << " Index" << OutPutInitialIndex << " 处理：" << double(end - start) / CLOCKS_PER_SEC <<"  总时间："<< double(end2 - start) / CLOCKS_PER_SEC<< endl;
					break;
				}
				int picture_index = OutPutInitialIndex + img_index;
				//Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.PictureNum;
				Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * img_index;
				//将图像数据传输到GPU
				for (int i = 0; i < GRAYCompressStreams; i++) {
					cudaMemcpy2DAsync(this->memory[i].d_bsrc, ImageSize.StrideF, gHostComressiongBuffer[mFlagIndex] + Bufferoffset,
						ImageSize.width * sizeof(unsigned char), ImageSize.width * sizeof(unsigned char), ImageSize.height,
						cudaMemcpyHostToDevice, this->stream[i]);
					Bufferoffset += gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.PictureNum;
				}
				this->process();
				this->writedisk(picture_index, &data_bag, img_index);
				img_index = img_index + gStructVarible.PictureNum * GRAYCompressStreams;
			}
		}
		delete[] total_malloc;
		this->memoryfree();
		for (int i = 0; i < GRAYCompressStreams; i++)
			//-------------销毁CUDA流
			cudaStreamDestroy(this->stream[i]);
	}
};
/*----------------------------------实现灰度图压缩功能的类--------------------------------*/
//npp库调用版
//class T : public Runnable									//由于该类的实现和前者TC类及其相似，所以不再进行注释
//{
//public:
//	HardwareInfo param;										//硬件参数
//	needmemory memory;
//	needdata staticdata;
//	static int mTindex;
//	static int test_number;
//	unsigned char* total_malloc;
//	int pix_index;
//
//
//public:
//	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
//	{
//		clock_t start_time, cur_time;
//		start_time = clock();
//		do
//		{
//			cur_time = clock();
//		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
//	}
//	void Initialize()
//	{
//		//cudaMalloc((void**)&(this->my_in), imgHeight * imgWidth * sizeof(unsigned char) * 3);
//		nppiDCTInitAlloc(&(this->memory).pDCTState);
//		cudaMalloc(&(this->staticdata).pdQuantizationTables, 64 * 4);
//
//		float nScaleFactor;
//		nScaleFactor = 1.0f;
//		int nMCUBlocksH = 0;
//		int nMCUBlocksV = 0;
//		quantityassgnment();
//
//		for (int i = 0; i < oFrameHeader.nComponents; ++i)
//		{
//			nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
//			nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
//		}
//
//
//		Npp8u aZigzag[] = {
//			0,  1,  5,  6, 14, 15, 27, 28,
//			2,  4,  7, 13, 16, 26, 29, 42,
//			3,  8, 12, 17, 25, 30, 41, 43,
//			9, 11, 18, 24, 31, 40, 44, 53,
//			10, 19, 23, 32, 39, 45, 52, 54,
//			20, 22, 33, 38, 46, 51, 55, 60,
//			21, 34, 37, 47, 50, 56, 59, 61,
//			35, 36, 48, 49, 57, 58, 62, 63
//		};
//
//		for (int i = 0; i < 4; ++i)
//		{
//			Npp8u temp[64];
//
//			for (int k = 0; k < 32; ++k)
//			{
//				temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k + 0]];
//				temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
//			}
//
//			cudaMemcpyAsync((unsigned char *)(this->staticdata).pdQuantizationTables + i * 64, temp, 64, cudaMemcpyHostToDevice);
//
//		}
//
//		float frameWidth = floor((float)oFrameHeader.nWidth * (float)nScaleFactor);
//		float frameHeight = floor((float)oFrameHeader.nHeight * (float)nScaleFactor);
//
//		(this->staticdata).oDstImageSize.width = (int)max(1.0f, frameWidth);
//		(this->staticdata).oDstImageSize.height = (int)max(1.0f, frameHeight);
//
//		size_t newPitch[3];
//		NppiSize oBlocks;
//
//
//		for (int i = 0; i < oFrameHeader.nComponents; ++i)
//		{
//			//NppiSize oBlocks;
//			NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] & 0x0f, oFrameHeader.aSamplingFactors[i] >> 4 };
//
//			oBlocks.width = (int)ceil(((this->staticdata).oDstImageSize.width + 7) / 8 *
//				static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
//			oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;
//
//			oBlocks.height = (int)ceil(((this->staticdata).oDstImageSize.height + 7) / 8 *
//				static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
//			oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;
//
//			(this->staticdata).aDstSize[i].width = oBlocks.width * 8;
//			(this->staticdata).aDstSize[i].height = oBlocks.height * 8;
//		}
//
//
//		// Scale to target image size
//		// Assume we only deal with 420 images.
//		int aSampleFactor[3] = { 1, 2, 2 };
//
//		(this->memory).nScanSize = (this->staticdata).oDstImageSize.width * (this->staticdata).oDstImageSize.height * 2;
//		(this->memory).nScanSize = (this->memory).nScanSize > (4 << 20) ? (this->memory).nScanSize : (4 << 20);
//		cudaMalloc(&(this->memory).pDScan, (this->memory).nScanSize);
//		nppiEncodeHuffmanGetSize((this->staticdata).aDstSize[0], 3, &(this->memory).nTempSize);
//		cudaMalloc(&(this->memory).pDJpegEncoderTemp, (this->memory).nTempSize);
//
//
//		for (int j = 0; j < 3; j++) {
//			size_t nPitch1;
//			cudaMallocPitch(&(this->memory).pDCT[j], &nPitch1, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height);
//			(this->memory).DCTStep[j] = static_cast<Npp32s>(nPitch1);
//			//NPP_CHECK_CUDA(cudaMallocPitch(&myImage1[j], &nPitch1, aSrcSize[j].width, aSrcSize[j].height));   原来
//			cudaMallocPitch(&(this->memory).pDImage[j], &nPitch1, (this->staticdata).aDstSize[j].width, (this->staticdata).aDstSize[j].height);
//			(this->memory).DImageStep[j] = static_cast<Npp32s>(nPitch1);
//			dataduiqi[j] = nPitch1;
//
//		}
//		for (int i = 0; i < 3; ++i)
//		{
//			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &(this->staticdata).apDHuffmanDCTable[i]);
//			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &(this->staticdata).apDHuffmanACTable[i]);
//		}
//
//		for (int iComponent = 0; iComponent < 2; ++iComponent)
//		{
//			(this->memory).hpCodesDC[iComponent] = pHuffmanDCTables[iComponent].aCodes;
//			(this->memory).hpCodesAC[iComponent] = pHuffmanACTables[iComponent].aCodes;
//			(this->memory).hpTableDC[iComponent] = pHuffmanDCTables[iComponent].aTable;
//			(this->memory).hpTableAC[iComponent] = pHuffmanACTables[iComponent].aTable;
//		}
//	}
//	void process()
//	{
//		compress_process_lock.lock();
//		nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW((this->memory).pDImage[0], (this->memory).DImageStep[0],
//			(this->memory).pDCT[0], (this->memory).DCTStep[0],
//			(this->staticdata).pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[0] * 64,
//			(this->staticdata).aDstSize[0],
//			(this->memory).pDCTState);
//		compress_process_lock.unlock();
//
//
//		nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R((this->memory).pDCT, (this->memory).DCTStep,
//			0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
//			(this->memory).pDScan, &(this->memory).nScanLength,
//			(this->memory).hpCodesDC, (this->memory).hpTableDC, (this->memory).hpCodesAC, (this->memory).hpTableAC,
//			(this->staticdata).apDHuffmanDCTable,
//			(this->staticdata).apDHuffmanACTable,
//			(this->staticdata).aDstSize,
//			(this->memory).pDJpegEncoderTemp);
//
//
//	}
//
//	//void writedisk(int picture_num, Package* a, int bag_index, unsigned char*total_malloc, int& pix_index)
//	void writedisk(int picture_num, Package* a, int bag_index)
//	{
//		unsigned char *pDstJpeg = new unsigned char[(this->memory).nScanSize];
//		unsigned char *pDstOutput = pDstJpeg;
//
//		oFrameHeader.nWidth = (this->staticdata).oDstImageSize.width;
//		oFrameHeader.nHeight = (this->staticdata).oDstImageSize.height;
//
//		writeMarker(0x0D8, pDstOutput);
//		writeJFIFTag(pDstOutput);
//		writeQuantizationTable(aQuantizationTables[0], pDstOutput);
//		writeQuantizationTable(aQuantizationTables[1], pDstOutput);
//		writeFrameHeader(oFrameHeader, pDstOutput);
//		writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
//		writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
//		writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
//		writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
//		writeScanHeader(oScanHeader, pDstOutput);
//
//		cudaMemcpy(pDstOutput, (this->memory).pDScan, (this->memory).nScanLength, cudaMemcpyDeviceToHost);
//		pDstOutput += (this->memory).nScanLength;
//
//		writeMarker(0x0D9, pDstOutput);
//
//		char szOutputFiler[100];
//		sprintf_s(szOutputFiler, "%s\\%d.jpg", gStructVarible.ImgSavePath, picture_num);
//		memcpy(total_malloc + pix_index, pDstJpeg, static_cast<int>(pDstOutput - pDstJpeg));
//		pix_index += static_cast<int>(pDstOutput - pDstJpeg);
//		a->Form_one_head(bag_index / gStructVarible.PictureNum, szOutputFiler, pDstOutput - pDstJpeg);
//
//		delete[] pDstJpeg;
//
//		//Write result to file.
//		//std::ofstream outputFile1(OutputFile, ios::out | ios::binary);
//		//outputFile1.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
//		//delete[] pDstJpeg;
//	}
//	void memoryfree()
//	{
//		//cudaFree(this->my_in);
//		for (int i = 0; i < 3; ++i)
//		{
//			cudaFree((this->memory).pDCT[i]);
//			cudaFree((this->memory).pDImage[i]);
//			nppiEncodeHuffmanSpecFree_JPEG((this->staticdata).apDHuffmanDCTable[i]);
//			nppiEncodeHuffmanSpecFree_JPEG((this->staticdata).apDHuffmanACTable[i]);
//		}
//		nppiDCTFree((this->memory).pDCTState);
//		cudaFree((this->memory).pDJpegEncoderTemp);
//		cudaFree((this->memory).pDScan);
//		cudaFree((this->staticdata).pdQuantizationTables);
//	}
//	~T() {}
//	void Run()
//	{
//		char ImgoutputPath[255];
//		total_malloc = new unsigned char[100000000];
//		pix_index = 0;
//		clock_t start, end, end2;
//		int img_index;//图像索引
//		int mFlagIndex = 0;
//		int OutPutInitialIndex = 0; //输出的Bin文件初始索引号
//		int Bufferoffset = 0;//缓冲区偏移量
//		bool DatafullFlag = false;//标志位：当为true的时候，表示该GPU对应的两个缓冲区中，至少有一个有有效数据。
//								  //测试读入图片是否成功------------------------------------------------------------------------------------------------------------
//		cv::Mat img1(5120, 5120, CV_8UC1);
//		cudaSetDevice((this->param).GpuId);
//		this->Initialize();
//		cudaMemcpy2D((this->memory).pDImage[1], dataduiqi[1], gpHudata, compress_imgWidth * sizeof(unsigned char), compress_imgWidth * sizeof(unsigned char), compress_imgHeight, cudaMemcpyHostToDevice);
//		cudaMemcpy2D((this->memory).pDImage[2], dataduiqi[2], gpHvdata, compress_imgWidth * sizeof(unsigned char), compress_imgWidth * sizeof(unsigned char), compress_imgHeight, cudaMemcpyHostToDevice);
//
//		for (int i = 1; i < 3; ++i)
//		{
//			nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW((this->memory).pDImage[i], (this->memory).DImageStep[i],
//				(this->memory).pDCT[i], (this->memory).DCTStep[i],
//				(this->staticdata).pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
//				(this->staticdata).aDstSize[i],
//				(this->memory).pDCTState);
//		}
//		cout << "T GPU ：" << param.GpuId << " initial success!" << endl;
//
//		while (!ExtractPointSuccess)
//		{
//			mydelay(0.01);
//			img_index = 0;//图像计数
//			Bufferoffset = 0;
//			//绑定数据
//			while (true)//这里需要改，锁不能用和提点一样
//			{
//				gComressReadDataLock.lock();
//				mTindex = mTindex % (HardwareParam.DeviceCount + 1);
//				if (gComressionBufferEmpty[mTindex] == false && gComressionBufferWorking[mTindex] == false)
//				{
//					//将页锁内存标志位置为工作状态--进行绑定
//					gComressionBufferWorking[mTindex] = true;
//					OutPutInitialIndex = gComressionBufferStartIndex[mTindex] * Bufferlength;//获取图像首索引
//					mFlagIndex = mTindex;
//					DatafullFlag = true;
//					mTindex++;
//					gComressReadDataLock.unlock();
//					break;
//				}
//				mTindex++;
//				gComressReadDataLock.unlock();
//				if (ExtractPointSuccess)
//					break;
//			}
//			start = clock();
//			sprintf_s(ImgoutputPath, "%s\\%d.bin", gStructVarible.ImgSavePath, OutPutInitialIndex);
//			//Package data_bag(ImgoutputPath, Bufferlength);
//			Package data_bag(ImgoutputPath, Bufferlength / gStructVarible.PictureNum);
//			//压缩pImg图片
//			while (DatafullFlag)
//			{
//				if (img_index >= Bufferlength)
//				{
//					end = clock();
//					gComressReadDataLock.lock();
//					gComressionBufferWorking[mFlagIndex] = false;
//					gComressReadDataLock.unlock();
//					gComressionBufferEmpty[mFlagIndex] = true;
//					DatafullFlag = false;
//
//					//写磁盘
//					compress_write_lock.lock();
//					data_bag.file.open(data_bag.Fname, ios::out | ios::binary);
//					//data_bag.Form_total_head();
//					data_bag.Form_total_head(compress_imgWidth, compress_imgHeight, gStructVarible.PictureNum, OutPutInitialIndex);
//					//cout << OutPutInitialIndex << endl;
//					data_bag.file.write(data_bag.head_cache, data_bag.head_bias);
//					data_bag.file.write(reinterpret_cast<const char *>(total_malloc), static_cast<int>(pix_index));
//					data_bag.file.close();
//					//data_bag.UnPack(data_bag.Fname);
//					compress_write_lock.unlock();
//					memset(total_malloc, 0, 100000000);
//					pix_index = 0;
//					end2 = clock();
//					cout << "T GPU ：" << param.GpuId << " Index" << OutPutInitialIndex << " 处理：" << double(end - start) / CLOCKS_PER_SEC << "  总时间：" << double(end2 - start) / CLOCKS_PER_SEC << endl;
//					break;
//				}
//				int picture_index = OutPutInitialIndex + img_index;
//				Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.PictureNum;
//
//				cudaMemcpy2D((this->memory).pDImage[0], dataduiqi[0], gHostComressiongBuffer[mFlagIndex] + Bufferoffset, compress_old_Width * sizeof(unsigned char), compress_old_Width * sizeof(unsigned char), compress_old_Height, cudaMemcpyHostToDevice);
//				this->process();
//				this->writedisk(picture_index, &data_bag, img_index);
//				//img_index++;
//				img_index = img_index + gStructVarible.PictureNum;
//			}
//		}
//		delete[] total_malloc;
//		this->memoryfree();
//	}
//};
int T::mTindex = 0;
/*----------------------------------数据更新类--------------------------------------------*/
class  ReadImg : public Runnable
{
public:
	bool ExtractPointWorkingFlag = false;//表示 提点在工作
	bool CompressionWorkingFlag = false;//表示  压缩在工作 
	Parameter Devpar;//变量传参	
	~ReadImg()
	{
	}
	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	void Run()
	{
		Devpar.ImgHeight = gStructVarible.ImgHeight;
		Devpar.ImgWidth = gStructVarible.ImgWidth;
		Devpar.PictureNum = gStructVarible.PictureNum;
		Devpar.ImgChannelNum = gStructVarible.ImgChannelNum;
		int mPageLockBufferIndex = 0;
		int mCompressionBufferindex = 0;
		bool  ExtractCopySuccess;
		bool  ComressionCopySuccess;
		//初始化标志位
		for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
		{
			//页锁
			PageLockBufferEmpty[i] = true;
			PageLockBufferWorking[i] = false;
			PageLockBufferStartIndex[i] = 0;
			//压缩
			gComressionBufferEmpty[i] = true;
			gComressionBufferWorking[i] = false;
			gComressionBufferStartIndex[i] = 0;
		}
		//cout << "ReadImg initial success!" << endl;
		while (!ExtractPointSuccess) // 实验结束的标志位
		{
			mydelay(0.01);
			for (int i = 0; i <HardwareParam.DeviceCount * 2; i++)//这个用于遍历相机对应buffer的标志位
			{
				ExtractCopySuccess = false;
				ComressionCopySuccess = false;
				if (CameraBufferFull[i]) //相机对应的内存是否有可用数据--当为true时,则表示相机对应内存的i号Buffer有可用数据
				{
					//拷贝方位盒数据至方位盒缓冲区
					if (gStructVarible.RecModelFlag == true && HostUpdateRec == false)
					{
						memcpy(gRecupImgData, gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgChannelNum*Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum);//这个是在内存区域拷贝图像
						HostUpdateRec = true;
					}

					//由相机内存拷贝数据到页锁内存
					if (ExtractPointWorkingFlag)
					{
						while (1) //遍历页锁Buffer,判断页锁内存缓冲区是否有空余Buffer块
						{
							mPageLockBufferIndex = mPageLockBufferIndex % (HardwareParam.DeviceCount + 1);
							if (PageLockBufferEmpty[mPageLockBufferIndex])//若某一页锁为空 
							{
								memcpy(gHostBuffer[mPageLockBufferIndex], gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgHeight*Devpar.ImgWidth *Devpar.ImgChannelNum* Bufferlength);//拷贝数据到页锁
								ExtractCopySuccess = true;
								PageLockBufferEmpty[mPageLockBufferIndex] = false;// 拷贝了数据之后将 标志位置为false;
								PageLockBufferStartIndex[mPageLockBufferIndex] = BufferBlockIndex[i];//索引拷贝
								mPageLockBufferIndex++;
								break;
							}
							mPageLockBufferIndex++;
							if (ExtractPointSuccess)
								break;
						}
					}
					else
						ExtractCopySuccess = true;

					// 由相机内存拷贝到压缩缓冲区
					if (CompressionWorkingFlag)
					{
						while (1) //遍历页锁Buffer,判断页锁内存缓冲区是否有空余Buffer块
						{
							mCompressionBufferindex = mCompressionBufferindex % (HardwareParam.DeviceCount + 1);
							if (gComressionBufferEmpty[mCompressionBufferindex])//若某一页锁为空 
							{
								memcpy(gHostComressiongBuffer[mCompressionBufferindex], gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgHeight*Devpar.ImgWidth *Devpar.ImgChannelNum* Bufferlength);//拷贝数据到页锁
								ComressionCopySuccess = true;
								gComressionBufferEmpty[mCompressionBufferindex] = false;// 拷贝了数据之后将 标志位置为false;
								gComressionBufferStartIndex[mCompressionBufferindex] = BufferBlockIndex[i];//索引拷贝
								mCompressionBufferindex++;
								break;
							}
							mCompressionBufferindex++;
							if (ExtractPointSuccess)
								break;
						}
					}
					else
						ComressionCopySuccess = true;

					//相机内存对应标志位置false
					if (ExtractCopySuccess&&ComressionCopySuccess)
						CameraBufferFull[i] = false;
				}
			}
		}

	}
};

/*----------------------------------模拟数据产生类----------------------------------------*/
class  DataRefresh : public Runnable
{
public:
	Parameter Devpar;//变量传参	
	~DataRefresh()
	{
	}
	void mydelay(double sec)//延时函数，用于图像数据缓冲区的更新
	{
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < sec);
	}
	void Run()
	{
		//初始化参数
		Devpar.ImgHeight = gStructVarible.ImgHeight;
		Devpar.ImgWidth = gStructVarible.ImgWidth;
		Devpar.PictureNum = gStructVarible.PictureNum;
		Devpar.ImgChannelNum = gStructVarible.ImgChannelNum;
		clock_t start, end;
		char  path[250];
		//读图像
		unsigned char *Img1 = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.ImgChannelNum];
		if (Devpar.ImgChannelNum == 1)
		{
			RmwRead8BitBmpFile2Img(gStructVarible.ImgReadPath, NULL, Img1, &Devpar.ImgWidth, &Devpar.ImgHeight);
		}
		else
		{
			RmwRead8BitBmpFile2Img(gStructVarible.ImgReadPath, Img1, NULL, &Devpar.ImgWidth, &Devpar.ImgHeight);
		}
		for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
		{
			for (long long j = 0; j < Bufferlength; j++)
			{
				memcpy(gCameraBuffer[i] + j* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum, Img1, Devpar.ImgWidth* Devpar.ImgHeight*Devpar.ImgChannelNum * sizeof(unsigned char));
			}
		}
		//初始化索引计数
		for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
		{
			BufferBlockIndex[i] = i - HardwareParam.DeviceCount * 2;
			CameraBufferFull[i] = false;
		}
		//cout << "DataRefresh initial success!" << endl;/*调试项*/
		mydelay(2);
		//cout << " start!" << endl;
		//模拟相机更新数据
		start = clock();
		for (int q = 0; q <3; q++)   //5
		{
			for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
			{

				BufferBlockIndex[i] += HardwareParam.DeviceCount * 2;
				//若提点速度过慢，则这里会打印;
				if (CameraBufferFull[i])
				{
					//cout << "speed is too slow!" << endl;
					SimulationSuccessFlaf = true;
					ExtractPointSuccess = true;
					break;
				}
				CameraBufferFull[i] = true;
				mydelay(Timedatarefresh);
			}
			if (ExtractPointSuccess)
				break;
		}
		end = clock();
		mydelay(3);
		ExtractPointSuccess = true;
		//cout << "实验:50张图片更新时间=" << Timedatarefresh << "  over" << endl;
		delete[] Img1;

	}
};


//--------------------------------------------结束------------------------------------------//

/*对外接口函数*/

//调用动态库第一步
//对硬件设备初始化，自适应分配
/*************************************************
函数名称: GetDiskSpaceInfo  //

函数描述: 函数用于获取输入路径所在位置磁盘剩余容量(GB)； //

输入参数：LPCWSTR pszDrive ：路径所在位置盘符，
.		  比如"D:\1.bmp"则为"D:\"；//

输出参数：空；//

返回值  : RemainingSpace(int型) -- 剩余磁盘容量(GB)//

其他说明: 函数只用在硬件初始化时调用一次；
.		  函数还根据输入的驱动器，获取磁盘
.		  总容量空闲空间、簇数量等磁盘信息
.		  等，未引出接口//

*************************************************/
int GetDiskSpaceInfo(LPCWSTR pszDrive)
{
	DWORD64 qwFreeBytesToCaller, qwTotalBytes, qwFreeBytes;
	DWORD dwSectPerClust, dwBytesPerSect, dwFreeClusters, dwTotalClusters;
	BOOL bResult;

	//使用GetDiskFreeSpaceEx获取磁盘信息并打印结果  
	bResult = GetDiskFreeSpaceEx(pszDrive,
		(PULARGE_INTEGER)&qwFreeBytesToCaller,
		(PULARGE_INTEGER)&qwTotalBytes,
		(PULARGE_INTEGER)&qwFreeBytes);

	//使用GetDiskFreeSpace获取磁盘信息并打印结果  
	bResult = GetDiskFreeSpace(pszDrive,
		&dwSectPerClust,
		&dwBytesPerSect,
		&dwFreeClusters,
		&dwTotalClusters);

	int RemainingSpace;
	if (bResult)
	{
		RemainingSpace = int((DWORD64)dwFreeClusters*
			(DWORD64)dwSectPerClust*(DWORD64)dwBytesPerSect >> 30);
	}
	return RemainingSpace;
}

/*************************************************
函数名称: HardwareInit  //

函数描述: 硬件初始化； //

输入参数：null//

输出参数：HardwareInfo *HardwareProp ： 硬件配置信息；//

返回值  : (int型) -- 初始化成功或失败标志//

其他说明: 函数在软件启动初始化时调用
.		  对系统使用硬件资源配置；//

*************************************************/
IMGSIMULATION_API int HardwareInit(HardwareInfo *HardwareProp)
{
	if (gWorkingGpuId.size() != 0)
		gWorkingGpuId.clear();
	cudaGetDeviceCount(&gDeviceCount);

	//公共信息放在结构体HardwareParam中
	HardwareParam.DeviceCount = 0;//GPU设备数清零
	HardwareParam.DiskRemainingSpace = GetDiskSpaceInfo(L"C:/pic");//C盘剩余空间
	if (HardwareParam.DiskRemainingSpace < DiskRemainingSpaceThreshold)//%%%暂时定为100G%%%
	{
		return 1;//磁盘存储空间不足
	}
	for (int i = 0; i<gDeviceCount-1; i++)
	{
		cudaDeviceProp DevProp;
		cudaGetDeviceProperties(&DevProp, i);
		HardwareProp->major = DevProp.major;
		HardwareProp->minor = DevProp.minor;
		if (DevProp.major > 5)//计算能力大于5时
		{
			gWorkingGpuId.push_back(i);
		}
	}
	if (HardwareParam.DeviceCount > 5 && HardwareParam.DeviceCount < 1)
	{
		return 2;//最多可同时支持5块GPU
	}
	HardwareParam.DeviceCount = gWorkingGpuId.size();//GPU设备数目
	HardwareProp->DeviceCount = HardwareParam.DeviceCount;
	HardwareParam.ExPointThreads = HardwareParam.DeviceCount;//提点线程数
	HardwareProp->ExPointThreads = HardwareParam.DeviceCount;
	HardwareParam.CompThreads = HardwareParam.DeviceCount;//压缩线程数
	HardwareProp->CompThreads = HardwareParam.DeviceCount;

	return 0;
}
//-------------------------------------------------------结束----------------------------------------//

/*************************************************
函数名称: Image_Pretreatment  //

函数描述: 函数用于图像预处理； //

输入参数：const char *path ：图像文件夹路径，
.		  const char *exten ： 图像格式，比如".bmp"；
.		  int ChooseMode : 预处理选项--1 图像读入内存
.									 --2 内存释放//

输出参数：gHostImage[i]批量存放图像数据的数组；//

返回值  : gHostPathImgNumber(int型) -- 路径下图像数量//

其他说明: 函数在调试时使用，用于图像的批处理；//

*************************************************/
IMGSIMULATION_API int Image_Pretreatment(const char *path, const char *exten, int ChooseMode)
{
	cv::Directory dir;
	string filepath(path);
	string fileexten(exten);

	vector<string> filenames = dir.GetListFiles(filepath, fileexten, false);

	if (filenames.size() == NULL)
	{
		return 0;
	}
	else
	{
		gHostPathImgNumber = filenames.size();
	}

	switch (ChooseMode)
	{
	case 1:
	{
		//图像预处理，从硬盘批量读入内存
#ifdef Pretreatment
		char strFilename[100];
		int mWidth;
		int mHeight;
		for (int i = 0; i < gHostPathImgNumber; i++)
		{
			sprintf_s(strFilename, "%s\\%d.bmp", path, i + 1); //将图片的路径名动态的写入到strFilename这个地址的内存空间
			checkCudaErrors(cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
			if (gStructVarible.ImgBitDeep == 24)
			{
				gHostColorImage[i] = new unsigned char[gStructVarible.ImgHeight * gStructVarible.ImgWidth * 3];
			}
			RmwRead8BitBmpFile2Img(strFilename, gHostColorImage[i], gHostImage[i], &mWidth, &mHeight);
		}
#endif // Pretreatment
		break;
	}
	case 2:
	{
		//批量内存释放
#ifdef Pretreatment
		for (int i = 0; i < gHostPathImgNumber; i++)
		{
			cudaFreeHost(gHostImage[i]);
			if (gStructVarible.ImgBitDeep == 24)
			{
				cudaFreeHost(gHostColorImage[i]);
			}
		}
#endif // Pretreatment
		break;
	}
	default:
		break;
	}
	return gHostPathImgNumber;
}

/*************************************************
函数名称: SimulationImageTest  //

函数描述: 测试原图仿真测试； //

输入参数：const char *path ：测试图像路径；//

输出参数：Infomation *Info ： 测试实验数据；//

返回值  : bool -- 实验成功标志位//

其他说明: 函数包含对整幅测试图在不同实验模式下
.		  实验性能的测试；
.		  测试包括：单提点测试、单压缩测试、提点压缩测试//

*************************************************/
IMGSIMULATION_API bool SimulationImageTest(const char *path, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	gHostPathImgNumber = 5;//测试图片复制数量
	Info->ImgProcessingNumbers = gHostPathImgNumber;
	/****  图片导入  ****/
	for (int i = 0; i < gHostPathImgNumber; i++)//为图片申请锁页内存
	{
		err = cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth  *gStructVarible.PictureNum * sizeof(unsigned char), cudaHostAllocDefault);
		if (gStructVarible.ImgBitDeep == 24)
		{
			err = cudaHostAlloc((void**)&gHostColorImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth  *gStructVarible.PictureNum * 3 * sizeof(unsigned char), cudaHostAllocDefault);
		}
	}
	int Picoffset = gStructVarible.ImgHeight * gStructVarible.ImgWidth;//单张灰度图片地址偏移量
	int PicoffsetColor = gStructVarible.ImgHeight * gStructVarible.ImgWidth * 3;//单张图片地址偏移量
	for (int i = 0; i < gHostPathImgNumber; i++)//读取图片
	{
		for (int j = 0; j < gStructVarible.PictureNum; j++)
		{
			RmwRead8BitBmpFile2Img(path, gHostColorImage[i] + j * PicoffsetColor, gHostImage[i] + j * Picoffset, &mWidth, &mHeight);
		}
	}
	if (gStructVarible.RecModelFlag == 1)
		GetImgBoxHost(path);//提取包围盒
	Info->DeviceCount = HardwareParam.DeviceCount;
	Info->CPUThreadCount = ExtractPointThreads;
	clock_t start, finish;
	float Difftime;//时间差
	float ImageSize;//图像尺寸
	int ImgChannel;//图像通道
	int ThreadID;

	/****  单提点测试 ****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	//提点线程数为GPU设备数
	pExecutor->Init(1, HardwareParam.ExPointThreads, 1);
	SIM *ExtractPoint = new SIM[HardwareParam.ExPointThreads];
	RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
	//RecS recs;

	if (gStructVarible.RecModelFlag == 0)//全图模式
	{
		start = clock(); //计时开始
		ThreadID = 0x01;//线程号
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			ExtractPoint[i].HardwarePar.DeviceID = i;
			ExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
			ExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
			ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
			sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
			sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
			sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
			ExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
			ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
			ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
			ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
			ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
			ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
			ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
			ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
			ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
			ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
			ExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;

			/**** 提取标志点过程 ****/
			pExecutor->Execute(&ExtractPoint[i], ThreadID);
			ThreadID = ThreadID << 1;
		}

		pExecutor->Terminate();//终止线程
		delete pExecutor;//删除线程池	
		finish = clock();//计时结束					 
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;//得到两次记录之间的时间差
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	else //矩形模式
	{
		start = clock(); //计时开始
		ThreadID = 0x01;//线程号
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			RecExtractPoint[i].HardwarePar.DeviceID = i;
			RecExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
			RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.DeviceCount;
			RecExtractPoint[i].HardwarePar.CUDAStreamNum = 5;
			//RecExtractPoint[i].Devpar.DataReadPath = "C:\\pic\\img_data";
			RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
			RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
			RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
			RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
			RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
			RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
			RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
			RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
			RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;

			/**** 提取标志点过程 ****/
			pExecutor->Execute(&RecExtractPoint[i], ThreadID);
			ThreadID = ThreadID << 1;
		}

		pExecutor->Terminate();
		delete pExecutor;
		finish = clock();//计时结束
						 //得到两次记录之间的时间差
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	///****  单压缩测试****/
	//CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
	//pExecutor1->Init(1, HardwareParam.CompThreads, 1);
	//T *Compression_grey = new T[HardwareParam.CompThreads];
	//TC *Compression = new TC[HardwareParam.CompThreads];

	//start = clock(); //计时开始
	//ThreadID = 0x01;//线程号重置
	//for (int i = 0; i < HardwareParam.ExPointThreads; i++)
	//{
	//	/**** 参数传入 ****/
	//	Compression_grey[i].param.DeviceID = i;
	//	Compression_grey[i].param.GpuId = gWorkingGpuId[i];
	//	Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;
	//	Compression[i].param.DeviceID = i;
	//	Compression[i].param.GpuId = gWorkingGpuId[i];
	//	Compression[i].param.CompThreads = HardwareParam.CompThreads;
	//	if (gStructVarible.ImgBitDeep == 8)
	//	{
	//		pExecutor1->Execute(&Compression_grey[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//	else if (gStructVarible.ImgBitDeep == 24)
	//	{
	//		pExecutor1->Execute(&Compression[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//}
	//pExecutor1->Terminate();
	//delete pExecutor1;
	//finish = clock();//计时结束
	//				 //得到两次记录之间的时间差
	//Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
	//Info->CompressionTimes = Difftime;
	//ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
	//ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
	//Info->CompressionSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;

	///****  提点与压缩同步测试****/
	//CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
	//pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads, 1);

	//start = clock(); //计时开始
	//ThreadID = 0x01;//线程号重置
	//if (gStructVarible.RecModelFlag == 0)//全图模式
	//{
	//	for (int i = 0; i < HardwareParam.ExPointThreads; i++)
	//	{
	//		pExecutor2->Execute(&ExtractPoint[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//}
	//else
	//{
	//	for (int i = 0; i < HardwareParam.ExPointThreads; i++)
	//	{
	//		pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//}
	//if (gStructVarible.ImgBitDeep == 8)
	//{
	//	for (int i = 0; i < HardwareParam.CompThreads; i++)
	//	{
	//		pExecutor2->Execute(&Compression_grey[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//}
	//else if (gStructVarible.ImgBitDeep == 24)
	//{
	//	for (int i = 0; i < HardwareParam.CompThreads; i++)
	//	{
	//		pExecutor2->Execute(&Compression[i], ThreadID);
	//		ThreadID = ThreadID << 1;
	//	}
	//}
	//pExecutor2->Terminate();
	//delete pExecutor2;
	//finish = clock();//计时结束
	//				 //得到两次记录之间的时间差
	//Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
	//Info->SynchronizeTimes = Difftime;
	//ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
	//ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
	//Info->SynchronizeSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	//释放内存
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		err = cudaFreeHost(gHostImage[i]);
		if (gStructVarible.ImgBitDeep == 24)
		{
			err = cudaFreeHost(gHostColorImage[i]);
		}
		if (err != cudaSuccess)
		{
			return false;
		}
	}
	return true;
}

IMGSIMULATION_API void  SimulationTestReport(const char *path, Infomation *Info)
{
	//测提点加压缩
	Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//单张图片大小
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/s时缓冲区刷新时间。
	SimulationSuccessFlaf = false;
	while (!SimulationSuccessFlaf)
	{
		if (Timedatarefresh > minTimeRefresh)
		{
			Timedatarefresh = Timedatarefresh - 0.05;
			continue;
		}
		for (int i = 0; i < 3; i++)
			ExtractPointInitialSuccessFlag[i] = false;
		ExtractPointSuccess = false;
		Timedatarefresh = Timedatarefresh - 0.05;
		OnlineImageRecExperiment(3 , Info);
		//每次实验之后，延时两秒
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < 2);
	}
	Memory_release();
	Timedatarefresh = Timedatarefresh + 0.05;
	if (Timedatarefresh > minTimeRefresh)
		Timedatarefresh = minTimeRefresh;
	Info->SynchronizeTimes = Timedatarefresh;
	Info->SynchronizeSpeed = SiglePicSize*Bufferlength / Timedatarefresh;
}


/****************************************仿真实验相关********************************************/
//qwe 仿真总函数
IMGSIMULATION_API bool SimulationExperient(int ChooseMode)
{
	clock_t start, finish;
	Infomation *Info;
	float Difftime;//时间差
	float ImageSize;//图像尺寸
	int ImgChannel;//图像通道
	int ThreadID;

	//cout << "设备数目:" << HardwareParam.DeviceCount << endl;

	switch (ChooseMode)
	{
	case 1://单提点
	{
		/****  单提点测试****/
		CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
		//int  ThreadsNum;
		//if (gStructVarible.RecModelFlag == true)//qwt
		//	ThreadsNum = HardwareParam.ExPointThreads + 3;
		//else
		//	ThreadsNum = HardwareParam.ExPointThreads + 2;
		pExecutor->Init(1, 10, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		RecUpData recupdate;
		ReadImg  readimg;
		DataRefresh  datarefresh;
		readimg.CompressionWorkingFlag = false;
		readimg.ExtractPointWorkingFlag = true;
		if (gStructVarible.RecModelFlag == false)//全图模式
		{
			ThreadID = 0x01;//线程号
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				ExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				ExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				ExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;

				/**** 提取标志点过程 ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();//终止线程
		}
		else //矩形模式
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			ThreadID = 0x01;//线程号
							/**** 提取标志点过程 ****/
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.DeviceCount;
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				RecExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				RecExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;
				/**** 提取标志点过程 ****/
				pExecutor->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			pExecutor->Execute(&recupdate, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();

			cout << "实验结束" << endl;
			delete pExecutor;
		}
		break;
	}
	case 2://单压缩
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, 10, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		DataRefresh  datarefresh;
		ReadImg  readimg;
		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = false;
		ThreadID = 0x01;//线程号重置
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.GpuId = gWorkingGpuId[i];
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;
			Compression[i].param.DeviceID = i;
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgChannelNum == 1)
			{
				pExecutor1->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgChannelNum == 3)
			{
				pExecutor1->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		pExecutor1->Execute(&datarefresh, ThreadID);
		ThreadID = ThreadID << 1;
		pExecutor1->Execute(&readimg, ThreadID);
		pExecutor1->Terminate();
		delete pExecutor1;
		break;
	}
	case 3://提点&压缩
	{
		CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
		pExecutor2->Init(1, 10, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		RecUpData recupdate;
		ReadImg  readimg;
		DataRefresh  datarefresh;
		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = true;
		ThreadID = 0x01;//线程号
		//提点线程
		if (gStructVarible.RecModelFlag == false)//全图模式
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				ExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				ExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				ExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;

				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //矩形模式
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.DeviceCount;
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				RecExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				RecExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;
				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
				pExecutor2->Execute(&recupdate, ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//压缩线程
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.GpuId = gWorkingGpuId[i];
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;

			Compression[i].param.DeviceID = i;
			Compression[i].param.GpuId = gWorkingGpuId[i];
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgChannelNum == 1)
			{
				pExecutor2->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgChannelNum == 3)
			{
				pExecutor2->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//数据生成+读图线程
		pExecutor2->Execute(&readimg, ThreadID);
		ThreadID = ThreadID << 1;
		pExecutor2->Execute(&datarefresh, ThreadID);
		pExecutor2->Terminate();
		delete pExecutor2;
		delete[] ExtractPoint;
		delete[] RecExtractPoint;
		delete[] Compression_grey;
		delete[] Compression;
		break;
	}
	default: return 1;
	}
	return 0;
}

//qwe 提点和压缩同步
IMGSIMULATION_API void  SimulationTestSynchronize(const char *path, Infomation *Info)
{

	//测提点加压缩
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//单张图片大小
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/s时缓冲区刷新时间。
	SimulationSuccessFlaf = false;
	while (!SimulationSuccessFlaf)
	{
		if (Timedatarefresh > minTimeRefresh)
		{
			Timedatarefresh = Timedatarefresh - 0.05;
			continue;
		}
		for (int i = 0; i < 3; i++)
			ExtractPointInitialSuccessFlag[i] = false;
		ExtractPointSuccess = false;
		Timedatarefresh = Timedatarefresh - 0.05;
		SimulationExperient(3);

		//每次实验之后，延时两秒
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < 2);
	}
	Memory_release();
	Timedatarefresh = Timedatarefresh + 0.05;
	if (Timedatarefresh > minTimeRefresh)
		Timedatarefresh = minTimeRefresh;
	Info->SynchronizeTimes = Timedatarefresh;
	Info->SynchronizeSpeed = SiglePicSize*Bufferlength / Timedatarefresh;
}

//qwe 单提点
IMGSIMULATION_API void  SimulationTestExtractPoint(const char *path, Infomation *Info)
{

	//测提点加压缩
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//单张图片大小
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/s时缓冲区刷新时间。
	SimulationSuccessFlaf = false;
	while (!SimulationSuccessFlaf)
	{
		if (Timedatarefresh > minTimeRefresh)
		{
			Timedatarefresh = Timedatarefresh - 0.05;
			continue;
		}
		for (int i = 0; i < 3; i++)
			ExtractPointInitialSuccessFlag[i] = false;
		ExtractPointSuccess = false;
		Timedatarefresh = Timedatarefresh - 0.1;
		SimulationExperient(1);
		//每次实验之后，延时两秒
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < 2);
	}
	Memory_release();
	Timedatarefresh = Timedatarefresh + 0.05;
	if (Timedatarefresh > minTimeRefresh)
		Timedatarefresh = minTimeRefresh;
	Info->SynchronizeTimes = Timedatarefresh;
	Info->SynchronizeSpeed = SiglePicSize*Bufferlength / Timedatarefresh;
}

//qwe 单压缩
IMGSIMULATION_API void  SimulationTestComression(const char *path, Infomation *Info)
{

	//测提点加压缩
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//单张图片大小
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/s时缓冲区刷新时间。
	SimulationSuccessFlaf = false;
	while (!SimulationSuccessFlaf)
	{
		if (Timedatarefresh > minTimeRefresh)
		{
			Timedatarefresh = Timedatarefresh - 0.05;
			continue;
		}
		for (int i = 0; i < 3; i++)
			ExtractPointInitialSuccessFlag[i] = false;
		ExtractPointSuccess = false;
		Timedatarefresh = Timedatarefresh - 0.1;
		SimulationExperient(2);
		//每次实验之后，延时两秒
		clock_t start_time, cur_time;
		start_time = clock();
		do
		{
			cur_time = clock();
		} while (double(cur_time - start_time) / CLOCKS_PER_SEC < 2);
	}
	Memory_release();
	Timedatarefresh = Timedatarefresh + 0.05;
	if (Timedatarefresh > minTimeRefresh)
		Timedatarefresh = minTimeRefresh;
	Info->SynchronizeTimes = Timedatarefresh;
	Info->SynchronizeSpeed = SiglePicSize*Bufferlength / Timedatarefresh;
}

/*---------------------------------------------------------------------------------------*/

/*************************************************
函数名称: OnlineImageExperiment  //

函数描述: 在线实验模块--全图模式； //

输入参数：const char *Imgpath ：在线实验图像路径；
.		  ChooseMode ：1 单提点
.					   2 单压缩
.					   3 提点&压缩//

输出参数：Infomation *Info ： 在线实验数据；//

返回值  : bool -- 实验成功标志位//

其他说明: 函数选择性的进行三种模式的在线实验
.		  ，具体模式通过界面设置参数选择//

*************************************************/
IMGSIMULATION_API bool OnlineImageExperiment(int ChooseMode, const char *Imgpath, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	clock_t start, finish;
	float Difftime;//时间差
	float ImageSize;//图像尺寸
	int ImgChannel;//图像通道
	int ThreadID;

	switch (ChooseMode)
	{
	case 1://单提点
	{
		/****  单提点测试****/
		CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
		pExecutor->Init(1, HardwareParam.ExPointThreads, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		if (gStructVarible.RecModelFlag == 0)//全图模式
		{
			start = clock(); //计时开始
			ThreadID = 0x01;//线程号
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				ExtractPoint[i].HardwarePar.CUDAStreamNum = 5;
				//ExtractPoint[i].Devpar.DataReadPath = "C:\\pic\\img_data";
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;

				/**** 提取标志点过程 ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Terminate();//终止线程
			delete pExecutor;//删除线程池	
			finish = clock();//计时结束
							 //得到两次记录之间的时间差
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		else //矩形模式
		{
			start = clock(); //计时开始
			ThreadID = 0x01;//线程号
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = 5;
				//RecExtractPoint[i].Devpar.DataReadPath = "C:\\pic\\img_data";
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;

				/**** 提取标志点过程 ****/
				pExecutor->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Terminate();
			delete pExecutor;
			finish = clock();//计时结束
							 //得到两次记录之间的时间差
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		break;
	}
	case 2://单压缩
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, HardwareParam.CompThreads, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];

		start = clock(); //计时开始
		ThreadID = 0x01;//线程号重置
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;
			Compression[i].param.DeviceID = i;
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgBitDeep == 8)
			{
				pExecutor1->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgBitDeep == 24)
			{
				pExecutor1->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		pExecutor1->Terminate();
		delete pExecutor1;
		finish = clock();//计时结束
						 //得到两次记录之间的时间差
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->CompressionTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->CompressionSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		break;
	}
	case 3://提点&压缩
	{
		CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
		pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];

		ThreadID = 0x01;//线程号
		start = clock(); //计时开始
		if (gStructVarible.RecModelFlag == 0)//全图模式
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				ExtractPoint[i].HardwarePar.CUDAStreamNum = 5;
				//ExtractPoint[i].Devpar.DataReadPath = "C:\\pic\\img_data";
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;

				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //矩形模式
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = 5;
				//RecExtractPoint[i].Devpar.DataReadPath = "C:\\pic\\img_data";
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;

				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;
			Compression[i].param.DeviceID = i;
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgBitDeep == 8)
			{
				pExecutor2->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgBitDeep == 24)
			{
				pExecutor2->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}

		pExecutor2->Terminate();
		delete pExecutor2;
		finish = clock();//计时结束
						 //得到两次记录之间的时间差
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->SynchronizeTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->SynchronizeSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		break;
	}
	default: return 1;
	}
	return 0;
}

/*************************************************
函数名称: OnlineImageExperiment  //

函数描述: 在线实验模块--矩形模式； //

输入参数：const char *Imgpath ：在线实验图像路径；
.		  ChooseMode ：1 单提点
.					   2 单压缩
.					   3 提点&压缩//

输出参数：Infomation *Info ： 在线实验数据；//

返回值  : bool -- 实验成功标志位//

其他说明: 函数选择性的进行三种模式的在线实验
.		  ，具体模式通过界面设置参数选择//

*************************************************/
IMGSIMULATION_API bool OnlineImageRecExperiment(int ChooseMode, Infomation *Info)
{
	clock_t start, finish;
	int mWidth, mHeight;
	float Difftime;//时间差
	float ImageSize;//图像尺寸
	int ImgChannel;//图像通道
	int ThreadID;

	switch (ChooseMode)
	{
	case 1://单提点
	{
		/****  单提点测试****/
		CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
		int  ThreadsNum;
		if (gStructVarible.RecModelFlag == 1)//qwt
			ThreadsNum = HardwareParam.ExPointThreads + 3;
		else
			ThreadsNum = HardwareParam.ExPointThreads + 2;
		pExecutor->Init(1, 10, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];

		RecUpData recupdate;
		ReadImg  readimg;
		DataRefresh  datarefresh;

		readimg.CompressionWorkingFlag = false;
		readimg.ExtractPointWorkingFlag = true;

		if (gStructVarible.RecModelFlag == 0)//全图模式
		{
			start = clock(); //计时开始
			ThreadID = 0x01;//线程号
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				ExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				//sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				//sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				//sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", "C:\\pic\\img_read");
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", "C:\\pic\\img_write");
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", "C:\\pic\\img_data");
				ExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				ExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;

				/**** 提取标志点过程 ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();//终止线程
			delete pExecutor;//删除线程池	
			finish = clock();//计时结束
							 //得到两次记录之间的时间差
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		else //矩形模式
		{
			ThreadID = 0x01;//线程号
			GetImgBoxHost(gStructVarible.ImgReadPath);
			/**** 提取标志点过程 ****/
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.DeviceCount;
				sprintf_s(RecExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(RecExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(RecExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				RecExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;

				pExecutor->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			pExecutor->Execute(&recupdate, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();

			//cout << "实验结束" << endl;
			delete pExecutor;
		}
		break;
	}
	case 2://单压缩
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, HardwareParam.CompThreads + 2, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		ReadImg  readimg;
		DataRefresh  datarefresh;

		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = false;

		ThreadID = 0x01;//线程号重置
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.GpuId = gWorkingGpuId[i];
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;
			Compression[i].param.DeviceID = i;
			Compression[i].param.GpuId = gWorkingGpuId[i];
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgBitDeep == 8)
			{
				pExecutor1->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgBitDeep == 24)
			{
				pExecutor1->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		pExecutor1->Execute(&datarefresh, ThreadID);
		ThreadID = ThreadID << 1;
		pExecutor1->Execute(&readimg, ThreadID);
		pExecutor1->Terminate();
		delete pExecutor1;
		break;
	}
	case 3://提点&压缩
	{
		CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
		
		//if (gStructVarible.RecModelFlag == 1)//qwt
		//	pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads+2, 1);
		//else
		//	pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads+1, 1);
		pExecutor2->Init(1, 10, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		RecUpData recupdate;
		ReadImg  readimg;
		DataRefresh  datarefresh;

		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = true;

		ThreadID = 0x01;//线程号
		if (gStructVarible.RecModelFlag == 0)//全图模式
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				ExtractPoint[i].HardwarePar.DeviceID = i;
				ExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				ExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				ExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				sprintf_s(ExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(ExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(ExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				ExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				ExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				ExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				ExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				ExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				ExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				ExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				ExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				ExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				ExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				ExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;
				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //矩形模式
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** 参数传入 ****/
				RecExtractPoint[i].HardwarePar.DeviceID = i;
				RecExtractPoint[i].HardwarePar.GpuId = gWorkingGpuId[i];
				RecExtractPoint[i].HardwarePar.CUDAStreamNum = CUDAStreams;
				RecExtractPoint[i].HardwarePar.DeviceCount = HardwareParam.ExPointThreads;
				sprintf_s(RecExtractPoint[i].Devpar.ImgReadPath, "%s", gStructVarible.ImgReadPath);
				sprintf_s(RecExtractPoint[i].Devpar.ImgSavePath, "%s", gStructVarible.ImgSavePath);
				sprintf_s(RecExtractPoint[i].Devpar.DataReadPath, "%s", gStructVarible.DataReadPath);
				RecExtractPoint[i].Devpar.ImgBitDeep = gStructVarible.ImgBitDeep;
				RecExtractPoint[i].Devpar.ImgHeight = gStructVarible.ImgHeight;
				RecExtractPoint[i].Devpar.ImgWidth = gStructVarible.ImgWidth;
				RecExtractPoint[i].Devpar.Threshold = gStructVarible.Threshold;
				RecExtractPoint[i].Devpar.LengthMin = gStructVarible.LengthMin;
				RecExtractPoint[i].Devpar.LengthMax = gStructVarible.LengthMax;
				RecExtractPoint[i].Devpar.AreaMin = gStructVarible.AreaMin;
				RecExtractPoint[i].Devpar.AreaMax = gStructVarible.AreaMax;
				RecExtractPoint[i].Devpar.PictureNum = gStructVarible.PictureNum;
				RecExtractPoint[i].Devpar.PicBlockSize = gStructVarible.PicBlockSize;
				RecExtractPoint[i].Devpar.ImgChannelNum = gStructVarible.ImgBitDeep / 8;
				/**** 提取标志点过程 ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
				pExecutor2->Execute(&recupdate, ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//压缩线程
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** 参数传入 ****/
			Compression_grey[i].param.DeviceID = i;
			Compression_grey[i].param.GpuId = gWorkingGpuId[i];
			Compression_grey[i].param.CompThreads = HardwareParam.CompThreads;

			Compression[i].param.DeviceID = i;
			Compression[i].param.GpuId = gWorkingGpuId[i];
			Compression[i].param.CompThreads = HardwareParam.CompThreads;
			if (gStructVarible.ImgChannelNum == 1)
			{
				pExecutor2->Execute(&Compression_grey[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			else if (gStructVarible.ImgChannelNum == 3)
			{
				pExecutor2->Execute(&Compression[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//数据生成+读图线程
		pExecutor2->Execute(&readimg, ThreadID);
		ThreadID = ThreadID << 1;
		pExecutor2->Execute(&datarefresh, ThreadID);
		pExecutor2->Terminate();
		delete pExecutor2;
		break;
	}
	default: return 1;
	}
	return 0;
}

/*************************************************
函数名称: OnlineImageRefresh  //

函数描述: 在线实验读取缓冲区图像； //

输入参数：空//

输出参数：空//

返回值  : 空//

其他说明: //

*************************************************/
IMGSIMULATION_API int OnlineImageRefresh(unsigned char *pImg)
{
	if (gCameraBuffer[0] == NULL)
		return 1;
	//pImg指针内存在界面端申请，大小为单张图像大小
	memcpy(pImg, gCameraBuffer[0], gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.ImgChannelNum * sizeof(unsigned char));
	return 0;
}

/*************************************************
函数名称: OfflineImageExperiment  //

函数描述: 离线实验模块； //

输入参数：const char *Imgpath ：离线实验图像路径；

输出参数：Infomation *Info ： 离线实验数据；//

返回值  : bool -- 实验成功标志位//

其他说明: 离线实验只是对单张图像的重算过程，
.		  没有图像压缩的步骤；//

*************************************************/
IMGSIMULATION_API bool OfflineImageExperiment(const char *Imgpath, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	char strFilename[100];
	clock_t start, finish;
	float Difftime;//时间差
	float ImageSize;//图像尺寸
	int ImgChannel;//图像通道

	for (int i = 0; i<5; i++)
	{
		sprintf_s(strFilename, "%s", Imgpath); //将图片的路径名动态的写入到strFilename这个地址的内存空间 
		cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth * sizeof(unsigned char), cudaHostAllocDefault);
		if (gStructVarible.ImgBitDeep == 24)
		{
			gHostColorImage[i] = new unsigned char[gStructVarible.ImgHeight * gStructVarible.ImgWidth * 3];
		}
		RmwRead8BitBmpFile2Img(strFilename, gHostColorImage[i], gHostImage[i], &mWidth, &mHeight);
	}
	gHostPathImgNumber = 5;//最低处理张数

						   /****  单提点测试****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	pExecutor->Init(1, 1, 1);
	R ExtractPoint;
	RecR RecExtractPoint;

	if (gStructVarible.RecModelFlag == 0)//全图模式
	{
		start = clock(); //计时开始
						 /**** 参数传入 ****/
		ExtractPoint.HardwarePar.DeviceID = 0;
		//ExtractPoint.Devpar.DataReadPath = "C:\\pic\\img_data";
		ExtractPoint.Devpar.ImgHeight = gStructVarible.ImgHeight;
		ExtractPoint.Devpar.ImgWidth = gStructVarible.ImgWidth;
		ExtractPoint.Devpar.Threshold = gStructVarible.Threshold;
		ExtractPoint.Devpar.LengthMin = gStructVarible.LengthMin;
		ExtractPoint.Devpar.LengthMax = gStructVarible.LengthMax;
		ExtractPoint.Devpar.AreaMin = gStructVarible.AreaMin;
		ExtractPoint.Devpar.AreaMax = gStructVarible.AreaMax;
		ExtractPoint.Devpar.PictureNum = gStructVarible.PictureNum;
		ExtractPoint.Devpar.PicBlockSize = gStructVarible.PicBlockSize;

		/**** 提取标志点过程 ****/
		pExecutor->Execute(&ExtractPoint, 0x01);

		pExecutor->Terminate();//终止线程
		delete pExecutor;//删除线程池	
		finish = clock();//计时结束
						 //得到两次记录之间的时间差
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	else //矩形模式
	{
		start = clock(); //计时开始
						 /**** 参数传入 ****/
		RecExtractPoint.HardwarePar.DeviceID = 0;
		//RecExtractPoint.Devpar.DataReadPath = "C:\\pic\\img_data";
		RecExtractPoint.Devpar.ImgHeight = gStructVarible.ImgHeight;
		RecExtractPoint.Devpar.ImgWidth = gStructVarible.ImgWidth;
		RecExtractPoint.Devpar.Threshold = gStructVarible.Threshold;
		RecExtractPoint.Devpar.LengthMin = gStructVarible.LengthMin;
		RecExtractPoint.Devpar.LengthMax = gStructVarible.LengthMax;
		RecExtractPoint.Devpar.AreaMin = gStructVarible.AreaMin;
		RecExtractPoint.Devpar.AreaMax = gStructVarible.AreaMax;
		RecExtractPoint.Devpar.PictureNum = gStructVarible.PictureNum;
		RecExtractPoint.Devpar.PicBlockSize = gStructVarible.PicBlockSize;

		/**** 提取标志点过程 ****/
		pExecutor->Execute(&RecExtractPoint, 0x01);

		pExecutor->Terminate();
		delete pExecutor;
		finish = clock();//计时结束
						 //得到两次记录之间的时间差
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//图像通道数
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}

	for (int i = 0; i<5; i++)
	{
		err = cudaFreeHost(gHostImage[i]);
		if (gStructVarible.ImgBitDeep == 24)
		{
			err = cudaFreeHost(gHostColorImage[i]);
		}
	}
	return 0;
}

/*************************************************
函数名称: SinglePictureExtractPoint  //

函数描述: 离线模式提点函数； //

输入参数：const char *Imgpath  离线时需要进行提点操作的图像路径
const char *outputPath 提取出的特征文件输出的路径      //
输出参数：无													//

返回值  : 无													//

其他说明: 函数输出的特征文件绝对路径名称为  outputPath\\OffLine.bin	  //
.

*************************************************/
IMGSIMULATION_API bool SinglePictureExtractPoint(const char *Imgpath, const char*outputPath)
{
	char strfilename[255];
	Parameter Devpar;
	Devpar.ImgHeight = gStructVarible.ImgHeight;
	Devpar.ImgWidth = gStructVarible.ImgWidth;
	Devpar.Threshold = gStructVarible.Threshold;
	Devpar.LengthMin = gStructVarible.LengthMin;
	Devpar.LengthMax = gStructVarible.LengthMax;
	Devpar.AreaMin = gStructVarible.AreaMin;
	Devpar.AreaMax = gStructVarible.AreaMax;
	Devpar.PictureNum = 1;
	Devpar.PicBlockSize = gStructVarible.PicBlockSize;
	Devpar.ImgChannelNum = gStructVarible.ImgChannelNum;
	Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;
	Devpar.ColThreadNum = (Devpar.ImgMakeborderWidth / Devpar.PicBlockSize + 127) / 128 * 128;
	Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / Devpar.PicBlockSize;

	// 线程配置定义
	dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
	dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);

	//读取图片
	unsigned char *tHostImage;
	cudaHostAlloc((void**)&tHostImage, Devpar.ImgHeight *  Devpar.ImgWidth *Devpar.ImgChannelNum *Devpar.PictureNum * sizeof(unsigned char), cudaHostAllocDefault);
	if (Devpar.ImgChannelNum == 1)
	{
		RmwRead8BitBmpFile2Img(Imgpath, NULL, tHostImage, &Devpar.ImgWidth, &Devpar.ImgHeight);
	}
	else
	{

		RmwRead8BitBmpFile2Img(Imgpath, tHostImage, NULL, &Devpar.ImgWidth, &Devpar.ImgHeight);

	}
	//-------------------------------------------------------------------------------------------------------------------------------
	unsigned char * tDevColorImage;
	unsigned char * tDevGrayImage;
	unsigned char * tDevpad;
	unsigned char * tDev2val;
	unsigned char * tDevcounter;
	cudaMalloc((void**)&tDevColorImage, sizeof(unsigned char)* Devpar.ImgWidth* Devpar.ImgHeight*Devpar.ImgChannelNum*Devpar.PictureNum);
	cudaMalloc((void**)&tDevGrayImage, sizeof(unsigned char)* Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum);
	cudaMalloc((void**)&tDevpad, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight*Devpar.PictureNum);
	cudaMalloc((void**)&tDev2val, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight*Devpar.PictureNum);
	cudaMalloc((void**)&tDevcounter, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight*Devpar.PictureNum);
	//设备端显存申请
	short *  tDevRecXLeft;
	short *  tDevRecYLeft;
	short *  tDevRecXRight;
	short *  tDevRecYRight;
	short  * tDevLength;
	short  * tDevArea;
	double  *tDevXpos;
	double  *tDevYpos;
	short   *tDevIndex;
	cudaMalloc((void**)&tDevRecXLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//方位盒 xmin
	cudaMalloc((void**)&tDevRecYLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
	cudaMalloc((void**)&tDevRecXRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
	cudaMalloc((void**)&tDevRecYRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
	cudaMalloc((void**)&tDevLength, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//设备端输出	周长
	cudaMalloc((void**)&tDevArea, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				面积
	cudaMalloc((void**)&tDevXpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				xpos
	cudaMalloc((void**)&tDevYpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				ypos
	cudaMalloc((void**)&tDevIndex, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//提取特征有效标志
																							//输出空间申请
	short *  tHostRecXLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecYLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecXRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecYRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostLength = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostArea = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	double *  tHostXpos = new double[Devpar.ColThreadNum*Devpar.RowThreadNum];
	double *  tHostYpos = new double[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostIndex = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];

	//核函数执行
	if (Devpar.ImgChannelNum == 1)
	{
		cudaMemcpy(tDevGrayImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice);
		//执行灰度化，二值化核函数程序
		GrayMakeBorder << <mGrid1, 128 >> > (tDevGrayImage, tDevpad, Devpar);
	}
	else
	{
		cudaMemcpy(tDevColorImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice);
		ColorMakeBorder << <mGrid1, 128 >> > (tDevColorImage, tDevpad, Devpar);
	}
	//执行灰度化，二值化核函数程序
	Binarization << <mGrid1, 128 >> > (tDevpad, tDev2val, tDevcounter, Devpar);
	//边界提取
	Dilation << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
	cudaMemcpy(tDev2val, tDevcounter, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice);
	Erosion << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
	//提取周长和包围盒
	GetCounter << <mGrid2, 128 >> > (tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, Devpar);//提取轮廓的函数
	SelectTrueBox << <mGrid2, 128 >> >(tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevIndex, Devpar);
	SelectNonRepeatBox << <mGrid2, 128 >> > (tDevRecXLeft, tDevRecYLeft, tDevIndex, Devpar);
	GetNonRepeatBox << <mGrid2, 128 >> >(tDevRecXLeft, tDevRecYLeft, tDevIndex, Devpar);
	GetInfo << <mGrid2, 128 >> > (tDevpad, tDevIndex, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevXpos, tDevYpos, tDevArea, Devpar);

	//拷贝输出结果
	cudaMemcpy(tHostLength, tDevLength, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(tHostArea, tDevArea, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(tHostXpos, tDevXpos, sizeof(double)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(tHostYpos, tDevYpos, sizeof(double)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(tHostIndex, tDevIndex, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
	vector<CircleInfo>myInfo;
	int mtempindex = 0;
	for (int j = 0; j <Devpar.ColThreadNum * Devpar.RowThreadNum; j++)
	{
		if (tHostIndex[j] != 0)
		{
			CircleInfo temp;
			mtempindex++;
			temp.index = (short)mtempindex;
			temp.length = tHostLength[j];
			temp.area = tHostArea[j];
			temp.xpos = tHostXpos[j];
			temp.ypos = tHostYpos[j];
			myInfo.push_back(temp);
		}
	}
	if (myInfo.size() > 0)
	{
		FILE* fp;
		sprintf(strfilename, "%s\\OffLine.bin", outputPath); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
		fp = fopen(strfilename, "wb");
		fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
		fclose(fp);
	}
	//释放内存
	cudaFreeHost(tHostImage);
	cudaFree(tDevRecXLeft);
	cudaFree(tDevRecYLeft);
	cudaFree(tDevRecXRight);
	cudaFree(tDevRecYRight);
	cudaFree(tDevLength);
	cudaFree(tDevArea);
	cudaFree(tDevXpos);
	cudaFree(tDevYpos);
	cudaFree(tDevIndex);
	cudaFree(tDevColorImage);
	cudaFree(tDevGrayImage);
	cudaFree(tDevpad);
	cudaFree(tDev2val);
	cudaFree(tDevcounter);
	delete[]tHostRecXLeft;
	delete[]tHostRecYLeft;
	delete[]tHostRecXRight;
	delete[] tHostRecYRight;
	delete[]tHostLength;
	delete[]tHostArea;
	delete[]tHostXpos;
	delete[]tHostYpos;
	delete[]tHostIndex;

	return 0;
}

/*************************************************
函数名称: DrawPointFlag  //

函数描述: 标志点重绘； //

输入参数：const char *pathBin ：标志点的特征信息文件；
.		  const char *pathImg ： 读入图像路径；//

输出参数：const char *pathWrite ： 写出图像路径；//

返回值  : 空//

其他说明: 函数把从图像中提取的标志点数据重新标记
.		  到图像原位置上，并输出标记后的图像
.		  标记形式为红色十字形式；//

*************************************************/
IMGSIMULATION_API void DrawPointFlag(const char *pathBin, const char *pathImg, const char *pathWrite)
{
	//读取特征
	FILE *fr;
	fr = fopen(pathBin, "rb");

	//获取文件大小
	fseek(fr, 0, SEEK_END);//设置文件指针stream的位置为文件结尾
	long lSize = ftell(fr);//获取数据长度
	rewind(fr);//设置文件指针stream的位置为给定流的文件开头

			   //开辟输出空间
	int FlagSize = lSize / sizeof(CircleInfo);
	CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*FlagSize);

	//读取文件数据
	fread(RInfo, sizeof(CircleInfo), FlagSize, fr);
	fclose(fr);

	//绘制标志点十字架
	Mat Img = imread(pathImg, IMREAD_COLOR);
	cv::Vec3b pflag(0, 0, 255);
	for (int i = 0; i < FlagSize; i++)
	{
		CircleInfo myinfo = RInfo[i];
		Img.at<Vec3b>(myinfo.xpos, myinfo.ypos) = pflag;

		if (myinfo.xpos - 3 >= 0)
		{
			Img.at<Vec3b>(myinfo.xpos - 1, myinfo.ypos) = pflag;
			Img.at<Vec3b>(myinfo.xpos - 2, myinfo.ypos) = pflag;
			Img.at<Vec3b>(myinfo.xpos - 3, myinfo.ypos) = pflag;
		}

		if (myinfo.xpos + 3 <= gStructVarible.ImgHeight)
		{
			Img.at<Vec3b>(myinfo.xpos + 1, myinfo.ypos) = pflag;
			Img.at<Vec3b>(myinfo.xpos + 2, myinfo.ypos) = pflag;
			Img.at<Vec3b>(myinfo.xpos + 3, myinfo.ypos) = pflag;
		}

		if (myinfo.xpos - 3 >= 0)
		{
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos - 1) = pflag;
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos - 2) = pflag;
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos - 3) = pflag;
		}

		if (myinfo.ypos + 3 <= gStructVarible.ImgWidth)
		{
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos + 1) = pflag;
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos + 2) = pflag;
			Img.at<Vec3b>(myinfo.xpos, myinfo.ypos + 3) = pflag;
		}
	}
	imwrite(pathWrite, Img);
	free(RInfo);
}

/*************************************************
函数名称: Memory_application  //

函数描述: 全局内存申请； //

输入参数：空//

输出参数：空//

返回值  : 空//

其他说明: 函数按图像尺寸申请所需的全局内存；//

*************************************************/
IMGSIMULATION_API void Memory_application()
{
	compress_old_Width = gStructVarible.ImgWidth;
	compress_old_Height = gStructVarible.ImgHeight * gStructVarible.PictureNum;
	//imgWidth = gStructVarible.ImgWidth;						//从前端界面设置获取图片长宽
	//imgHeight = gStructVarible.ImgHeight * gStructVarible.PictureNum;
	compress_imgWidth = (compress_old_Width + 7) / 8 * 8;
	compress_imgHeight = (compress_old_Height + 7) / 8 * 8;
	//从这里开始压缩注释
	compressratio = gStructVarible.CompressionRatio;		//从前端界面设置获取压缩比
	int bmpSize = compress_imgWidth * compress_imgHeight;
	gpHudata = new unsigned char[bmpSize];					//灰度图片的色差值是确定的，提前设置好
	gpHvdata = new unsigned char[bmpSize];
	memset(gpHudata, 128, compress_imgHeight * compress_imgWidth);
	memset(gpHvdata, 128, compress_imgHeight * compress_imgWidth);
	blocks.x = compress_imgWidth / 8;								//设置cuda压缩程序的blocks为(imgWidth / 8,imgHeight / 8)
	blocks.y = compress_imgHeight / 8;
	blocks.z = 1;
	quantityassgnment();									//初始化主机端的全局变量


	/*申请数据缓冲区*/
	//相机采集卡所对应内存
	gCameraDress= (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength* HardwareParam.DeviceCount * 2);
	for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
	{
		gCameraBuffer[i] = gCameraDress + i*gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength;

	}
	//压缩缓冲区
	for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
	{
		gHostComressiongBuffer[i] = (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength);
	}

	//缓冲区页锁内存
	for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
	{
		cudaHostAlloc((void**)&gHostBuffer[i], gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char)*Bufferlength, cudaHostAllocDefault);
	}
	//矩形盒数据内存
	gRecupImgData = (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.PictureNum*gStructVarible.ImgChannelNum * sizeof(unsigned char));
}

/*************************************************
函数名称: Memory_release  //

函数描述: 全局内存释放； //

输入参数：空//

输出参数：空//

返回值  : 空//

其他说明: 函数按图像尺寸释放所需的全局内存；//

*************************************************/
IMGSIMULATION_API void Memory_release()
{
	free(gCameraDress);
	gCameraDress = NULL;
	for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
	{
		//free(gCameraBuffer[i]);
		gCameraBuffer[i] = NULL;
	}

	for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
	{
		cudaFreeHost(gHostBuffer[i]);
		free(gHostComressiongBuffer[i]);
		gHostComressiongBuffer[i] = NULL;
	}

	free(gRecupImgData);
	delete[]gpHudata;
	delete[]gpHvdata;//qwt这里要出现错误
}

/*************************************************
函数名称: SetCameraPar  //

函数描述: 相机参数设置； //

输入参数：int ScrBufferlength ： 图片缓冲区的长度（图片张数）；
.

输出参数：null；//

返回值  : bool -- 参数传递成功标志位//

其他说明: 函数用于将界面设置的参数导入DLL的参数中；//

*************************************************/
IMGSIMULATION_API bool SetCameraPar(int ScrBufferlength)
{
	Bufferlength = ScrBufferlength;
	return 0;
}

/*************************************************
函数名称: SetParameter  //

函数描述: 参数传递； //

输入参数：Parameter *info ： 界面设置的结构体参数；
.		  int len ： 待传递参数个数；//

输出参数：Parameter gStructVarible ： 运行时的结构体参数；//

返回值  : bool -- 参数传递成功标志位//

其他说明: 函数用于将界面设置的参数导入DLL的参数中；//

*************************************************/
IMGSIMULATION_API bool SetParameter(Parameter *info, int len)
{
	char count = 0;

	if (info->ImgReadPath != NULL)
	{
		//gStructVarible.ImgReadPath = info->ImgReadPath;
		sprintf_s(gStructVarible.ImgReadPath, "%s//1.bmp", info->ImgReadPath);
		count++;
	}
	if (info->ImgSavePath != NULL)
	{
		//gStructVarible.ImgSavePath = info->ImgSavePath;
		sprintf_s(gStructVarible.ImgSavePath, "%s", info->ImgSavePath);
		count++;
	}
	if (info->DataReadPath != NULL)
	{
		//gStructVarible.DataReadPath = info->DataReadPath;
		sprintf_s(gStructVarible.DataReadPath, "%s", info->DataReadPath);
		count++;
	}
	if (info->ImgBitDeep != -1)
	{
		gStructVarible.ImgBitDeep = info->ImgBitDeep;
		count++;
	}
	if (info->ImgChannelNum != -1)
	{
		gStructVarible.ImgChannelNum = info->ImgChannelNum;
		count++;
	}
	if (info->ImgHeight != -1)
	{
		gStructVarible.ImgHeight = info->ImgHeight;
		count++;
	}
	if (info->ImgWidth != -1)
	{
		gStructVarible.ImgWidth = info->ImgWidth;
		count++;
	}
	if (info->Threshold != -1)
	{
		gStructVarible.Threshold = info->Threshold;
		count++;
	}
	if (info->LengthMin != -1)
	{
		gStructVarible.LengthMin = info->LengthMin;
		count++;
	}
	if (info->LengthMax != -1)
	{
		gStructVarible.LengthMax = info->LengthMax;
		count++;
	}
	if (info->PicBlockSize != -1)
	{
		gStructVarible.PicBlockSize = info->PicBlockSize;
		count++;
	}
	if (info->AreaMin != -1)
	{
		gStructVarible.AreaMin = info->AreaMin;
		count++;
	}
	if (info->AreaMax != -1)
	{
		gStructVarible.AreaMax = info->AreaMax;
		count++;
	}
	if (info->CompressionRatio != -1)
	{
		gStructVarible.CompressionRatio = info->CompressionRatio;
		count++;
	}
	if (info->PictureNum != -1)
	{
		gStructVarible.PictureNum = info->PictureNum;
		count++;
	}
	if (info->TerminateFlag != -1)
	{
		gStructVarible.TerminateFlag = info->TerminateFlag;
		if (gStructVarible.TerminateFlag == 1)
		{
			ExtractPointSuccess = true;//实验结束
		}
		else
		{
			ExtractPointSuccess = false;//标志位复位
		}
		count++;
	}
	if (info->RecModelFlag != -1)
	{
		gStructVarible.RecModelFlag = info->RecModelFlag;
		count++;
	}
	if (info->RecPadding != -1)
	{
		gStructVarible.RecPadding = info->RecPadding;
		count++;
	}
	gStructVarible.ImgChannelNum = gStructVarible.ImgBitDeep / 8;//通道数=位深度/8
	//传参个数校验
	if (count == len)
		return true;
	return false;
}

/*************************************************
函数名称: GetParameter  //

函数描述: 获取界面传入参数结构体参数值； //

输入参数：Parameter *info ： 界面传入参数结构体；

输出参数：Parameter gStructVarible ： 运行时的结构体参数；//

返回值  : NULL //

其他说明: 函数用于读取界面传入参数结构体参数；//

*************************************************/
IMGSIMULATION_API void GetParameter(Parameter *info)
{
	/*>1<*/sprintf_s(info->ImgReadPath, "%s", gStructVarible.ImgReadPath);
	/*>2<*/sprintf_s(info->ImgSavePath, "%s", gStructVarible.ImgSavePath);
	/*>3<*/sprintf_s(info->DataReadPath, "%s", gStructVarible.DataReadPath);
	/*>4<*/info->ImgBitDeep = gStructVarible.ImgBitDeep;
	/*>5<*/info->ImgChannelNum = gStructVarible.ImgChannelNum;
	/*>6<*/info->ImgHeight = gStructVarible.ImgHeight;
	/*>7<*/info->ImgWidth = gStructVarible.ImgWidth;
	/*>8<*/info->ImgMakeborderWidth = gStructVarible.ImgMakeborderWidth;
	/*>9<*/info->Threshold = gStructVarible.Threshold;
	/*>10<*/info->LengthMin = gStructVarible.LengthMin;
	/*>11<*/info->LengthMax = gStructVarible.LengthMax;
	/*>12<*/info->PicBlockSize = gStructVarible.PicBlockSize;
	/*>13<*/info->ColThreadNum = gStructVarible.ColThreadNum;
	/*>14<*/info->RowThreadNum = gStructVarible.RowThreadNum;
	/*>15<*/info->AreaMin = gStructVarible.AreaMin;
	/*>16<*/info->AreaMax = gStructVarible.AreaMax;
	/*>17<*/info->CompressionRatio = gStructVarible.CompressionRatio;
	/*>18<*/info->PictureNum = gStructVarible.PictureNum;
	/*>19<*/info->TerminateFlag = gStructVarible.TerminateFlag;
	/*>20<*/info->RecModelFlag = gStructVarible.RecModelFlag;
	/*>21<*/info->RecPadding = gStructVarible.RecPadding;
}

/*************************************************
函数名称: ClearDataCache  //

函数描述:  缓存清空函数；将DLL库中的全局变量、静态变量等全部设置成为初始化状态 //

输入参数：无					 //
输出参数：无					//

返回值  : 无					//

其他说明: 无 //
.

*************************************************/
IMGSIMULATION_API void  ClearDataCache()
{
	//全局变量情况
	sprintf_s(gStructVarible.ImgReadPath, "%s//1.bmp", "C://pic//img_read");
	sprintf_s(gStructVarible.ImgSavePath, "%s", "C://pic//img_write");
	sprintf_s(gStructVarible.DataReadPath, "%s", "C://pic//img_data");
	gStructVarible.AreaMax = 99999;
	gStructVarible.AreaMin = 0;
	gStructVarible.ColThreadNum = 320;
	gStructVarible.CompressionRatio = 2000;
	gStructVarible.ImgChannelNum = 1;
	gStructVarible.ImgHeight = 5120;
	gStructVarible.ImgMakeborderWidth = 5120;
	gStructVarible.ImgWidth = 5120;
	gStructVarible.LengthMax = 99999;
	gStructVarible.LengthMin = 0;
	gStructVarible.PicBlockSize = 16;
	gStructVarible.PictureNum = 1;
	gStructVarible.RecModelFlag = false;
	gStructVarible.RecPadding = 4;
	gStructVarible.RowThreadNum = 320;
	gStructVarible.Threshold = 60;
	gStructVarible.TerminateFlag = 0;

	//类静态变量
	R::mRindex = 0;
	RecR::mRecindex = 0;
	T::mTindex = 0;
	TC::mTCindex = 0;

	//全局
	Bufferlength = 50;
	ExtractPointInitialSuccessFlag[0] = false;
	ExtractPointInitialSuccessFlag[1] = false;
	ExtractPointInitialSuccessFlag[2] = false;
	ExtractPointSuccess = false;

	//矩形盒数据
	gHostRecData.clear();
	gRecNum = gHostRecData.size();
	gSingleImgRecNum = gHostRecData.size();
	gRecupImgData = NULL;
	DevUpdateRec[0] = false;
	DevUpdateRec[1] = false;
	DevUpdateRec[2] = false;
	HostUpdateRec = false;
	RecupdataInitialSuccessFlag = false;


	//相机缓冲
	for (int i = 0; i < 6; i++)
	{
		BufferBlockIndex[i] = 0;
		gCameraBuffer[i] = false;
		CameraBufferFull[i] = false;
	}

	for (int i = 0; i<4; i++)
	{
		//页锁内存缓冲区
		gHostBuffer[i] = NULL;
		PageLockBufferEmpty[i] = true;
		PageLockBufferWorking[i] = false;
		PageLockBufferStartIndex[i] = 0;

		//压缩缓冲
		gHostComressiongBuffer[i] = NULL;
		gComressionBufferEmpty[i] = true;
		gComressionBufferWorking[i] = false;
		gComressionBufferStartIndex[i] = 0;

	}

}


/****************************解压特征文件***************************************************/
/*************************************************
函数名称: GetFiles  //

函数描述: 遍历某一文件夹内的所有特征文件的路径；
函数将一个包含多张图像特征的特征文件(.bin文件)分解为多个特征文件(.bin),每张图片对应一个特征文件 //

输入参数：const char * path 包含多个.bin文件的文件夹路径

输出参数： vector<string>& files ；将文件夹之内的bin文件路径以string方式存起来	   //

返回值  : 无													//

其他说明: 无 //
.

*************************************************/
IMGSIMULATION_API void GetFiles(const char * path, vector<string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	//文件信息，声明一个存储文件信息的结构体  
	struct __finddata64_t fileinfo;
	string p;//字符串，存放路径
	if ((hFile = _findfirst64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
	{
		do
		{
			//如果是目录,迭代之（即文件夹内还有文件夹）  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				//文件名不等于"."&&文件名不等于".."
				//.表示当前目录
				//..表示当前目录的父目录
				//判断时，两者都要忽略，不然就无限递归跳不出去了！
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					GetFiles(p.assign(path).append("\\").append(fileinfo.name).c_str(), files);
			}
			//如果不是,加入列表  
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext64(hFile, &fileinfo) == 0);
		//_findclose函数结束查找
		_findclose(hFile);
	}
}

/*************************************************
函数名称: UnzipFeatureBins  //

函数描述: 解包函数；函数将一个包含多张图像特征的特征文件(.bin文件)分解为多个特征文件(.bin),每张图片对应一个特征文件 //

输入参数： const char *InputPath  包含多张图片特征的特征文件（50张，跟bufferlenth有关）
const char *OutputFilename 输出特征文件的文件夹      //
输出参数：无													//

返回值  : 无													//

其他说明: 无 //
.

*************************************************/
void UnzipFeatureBins(const char *InputPath, const char *OutputFilename)
{
	char strFilename[255];
	FILE *fr;
	fr = fopen(InputPath, "rb");
	if (fr == NULL)//若图像打不开，则return
	{
		cout << "FILE fail open" << endl;
		return;
	}
	fseek(fr, 0, SEEK_END);
	long lSize = ftell(fr);//获取数据长度
	rewind(fr);
	int Datalength = lSize / sizeof(CircleInfo);
	CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*Datalength);
	fread(RInfo, sizeof(CircleInfo), Datalength, fr);
	fclose(fr);
	//获取数据总个数
	int Dataoffset = 0;
	int Dataindex = 0;
	while (Dataoffset < Datalength)
	{
		CircleInfo mHead = RInfo[Dataoffset];
		Dataoffset++;
		int  mlen = 0;
		if (mHead.area == 0 && int(mHead.xpos) == 99999)//判断头特征
		{
			Dataindex = mHead.index;
			mlen = mHead.length;
			if (mlen > 0 && Dataoffset + mlen <= Datalength)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", OutputFilename, Dataindex); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
				fp = fopen(strFilename, "wb");
				fwrite(&RInfo[Dataoffset], sizeof(CircleInfo)*mlen, 1, fp);
				fclose(fp);
				Dataoffset = Dataoffset + mlen;
			}
		}
	}
};

/*************************************************
函数名称: UnzipFeatureFiles  //

函数描述: 解包函数；将某一个文件夹下的大特征文件（包含多张图片特征的文件.bin文件）解压成单个特征文件。
在GPU端提点所生成的特征文件是一个特征文件包含bufferlenth张图的特征，该函数将这个大特征文件
分解成多个bufferlenth个图片。

输入参数： const char *Filepath  包含特征文件的的文件夹

输出参数：无													//

返回值  : 无													//

其他说明: 无 //
.

*************************************************/
IMGSIMULATION_API void UnzipFeatureFiles(const char * Filepath)
{
	vector<string>FeatureFilesPass;
	GetFiles(Filepath, FeatureFilesPass);
	if (FeatureFilesPass.size() > 0)
	{
		for (int i = 0; i < FeatureFilesPass.size(); i++)
		{
			UnzipFeatureBins(FeatureFilesPass[i].c_str(), Filepath);
		}
	}
}

//解压图片函数
void UnzipOneBin(const char* Filepath, const char* BinPath)
{
	//Package temp(BinPath, content);
	Package temp(BinPath);
	temp.UnPack(BinPath, Filepath);
	return;
}

IMGSIMULATION_API void UnzipPictureFiles(const char * Filepath)
{
	vector<string>FeatureFilesPass;
	GetFiles(Filepath, FeatureFilesPass);
	int interpret;
	if (FeatureFilesPass.size() > 0)
	{
		for (int i = 0; i + 5 < FeatureFilesPass.size(); i = i + 5)
		{
			//cout << FeatureFilesPass[i].c_str() << endl;
			//Package cc(FeatureFilesPass[i].c_str(), Bufferlength/ gStructVarible.PictureNum);
			//cc.UnPack(FeatureFilesPass[i].c_str(), Filepath);
			//thread th1(UnzipOneBin, Filepath, FeatureFilesPass[i].c_str(), Bufferlength / gStructVarible.PictureNum);
			//thread th2(UnzipOneBin, Filepath, FeatureFilesPass[i+1].c_str(), Bufferlength / gStructVarible.PictureNum);
			//thread th3(UnzipOneBin, Filepath, FeatureFilesPass[i+2].c_str(), Bufferlength / gStructVarible.PictureNum);
			//thread th4(UnzipOneBin, Filepath, FeatureFilesPass[i+3].c_str(), Bufferlength / gStructVarible.PictureNum);
			//thread th5(UnzipOneBin, Filepath, FeatureFilesPass[i+4].c_str(), Bufferlength / gStructVarible.PictureNum);
			thread th1(UnzipOneBin, Filepath, FeatureFilesPass[i].c_str());
			thread th2(UnzipOneBin, Filepath, FeatureFilesPass[i + 1].c_str());
			thread th3(UnzipOneBin, Filepath, FeatureFilesPass[i + 2].c_str());
			thread th4(UnzipOneBin, Filepath, FeatureFilesPass[i + 3].c_str());
			thread th5(UnzipOneBin, Filepath, FeatureFilesPass[i + 4].c_str());
			interpret = i + 5;
			th1.join();
			th2.join();
			th3.join();
			th4.join();
			th5.join();
		}
		for (; interpret < FeatureFilesPass.size(); interpret++)
		{
			//Package cc(FeatureFilesPass[interpret].c_str(), Bufferlength / gStructVarible.PictureNum);
			Package cc(FeatureFilesPass[interpret].c_str());
			cc.UnPack(FeatureFilesPass[interpret].c_str(), Filepath);
		}
	}
}

/*---------------------------------------------------------------------------------------*/



