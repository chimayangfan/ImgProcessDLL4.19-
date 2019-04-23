#include"Imgsimulation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string.h>
#include <fstream> 
#include <string>
#include <io.h>
#include <vector>
#include <time.h>  

#include <stdio.h>  
#include<algorithm>

#include "Thread.h"
#include "ThreadPoolExecutor.h"
#include"cuda_profiler_api.h"
#include <helper_cuda.h>//错误处理
#include <Windows.h>
#include <GdiPlus.h>

#include <helper_string.h>
#include <npp.h>

#pragma comment( lib, "GdiPlus.lib" )
using namespace Gdiplus;
using namespace std;
using namespace cv;

//根据设备性能定义
#define ExtractPointThreads 1
#define CUDAStreams 2
int gHostImgblock = ExtractPointThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
//根据图片大小定义block和thread个数 
Parameter gStructVarible{NULL,NULL,NULL,8,5120,5120,5120,60,30,300,640,640,0,9999,1,false};
Infomation SignPoint;

#define Pretreatment
	#ifdef Pretreatment
	#define ReadImageNumber 250
#endif // Pretreatment
unsigned char* gHostImage[250] = { NULL };
unsigned char* gHostColorImage[250] = { NULL };

unsigned char* rhost_in[CUDAStreams];//页锁定内存
unsigned char* rDev_in[CUDAStreams];//设备内存
unsigned char* rDev_padding[CUDAStreams];//填充边界后的图像内存   qwt7.26
unsigned char* rgpu_2val[CUDAStreams];//二值化图
unsigned char* rgpu_counter[CUDAStreams];//轮廓图，在执行findcountores之后才生成

cudaStream_t *rcS;


//-------------------------方位盒Model数据-----------------------------//
typedef struct
{
	short RecXmin;
	short RecYmin;
	short RecXmax;
	short RecYmax;
}RecData;//方位盒数据结构
vector<RecData> gHostRecData;//CPU方位盒数据容器
int gRectRealNum;//方位盒的实际数量
//方位盒更新数据
int gImgXcenter;//图片灰度中心点（用所得的灰度中心加权平均）
int gImgYcenter;
int gXcenterOffset;//包围盒偏移量(包围盒更新时所用的值)
int gYcenterOffset;
//特征数据
struct CircleInfo
{
	short index;
	short length;
	short area;
	short xpos;
	short ypos;
};
//-------------------------------------------------------结束----------------------------------------//

/*------------------------------------------------核函数--------------------------------------------------*/
//--------------------------------------------------------开始---------------------------------------------//

/*填充图像边界*/
//输入为原图图像、图像高度、图像宽度  输出为填充后的宽度  填充后的宽度计算公式   int imgWidth = (width + 127) / 128 * 128;
__global__ void  CopyMakeBorder(const unsigned char *src, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//Id_y表示图像列索引
	const int Id_x = blockIdx.y;
	if (Id_y <  devpar.ImgWidth)
	{
		dst[Id_y + Id_x * devpar.ImgMakeborderWidth] = src[Id_y + Id_x * devpar.ImgWidth];
	}
}

/*二值化*/
__global__ void Binarization(unsigned char *psrcgray, unsigned char *pdst2val, unsigned char *pdstcounter, Parameter devpar)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//【二维grid和一维block】
	int temp = int(psrcgray[Id]);//寄存器保存像素，提高访存效率								
	if (Id < devpar.ImgMakeborderWidth * devpar.ImgHeight*devpar.PictureNum)
	{
		pdst2val[Id] = unsigned char(255 * int(temp>devpar.Threshold));//二值化，利用计算代替分支结构
		pdstcounter[Id] = unsigned char(255 * int(temp>devpar.Threshold));
	}
}

/*获取轮廓（边缘检测）*/
//膨胀
__global__  void Dilation(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_x代表行信息  Id_y代表列信息
	const int Id_x = blockIdx.y;
	int temp;
	if (Id_y> 0 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x <devpar.ImgHeight*devpar.PictureNum - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] == 0)
		{
			temp = int(psrc[Id_y - 1 + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x - 1)* devpar.ImgMakeborderWidth])
				+ int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y - 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]);
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp > 0 ? 255 : 0;
		}
	}

}

//腐蚀
__global__  void Erosion(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y代表行信息  Id_x代表列信息
	const int Id_x = blockIdx.y;//Id_x代表列信息
	int temp;
	//利用4领域值掏空内部点，提取轮廓信息，现在的dst就是存储轮廓的信息
	if (Id_y > 0 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x <devpar.ImgHeight*devpar.PictureNum - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] != 0)
		{
			temp = int(psrc[Id_y + (Id_x - 1)*devpar.ImgMakeborderWidth]) + int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)*devpar.ImgMakeborderWidth]);//用4领域腐蚀
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp >= 1020 ? 0 : 255;
		}
	}
}

/*轮廓提取*/
//src为轮廓数组（边缘检测结果），c_length提取的周长值，   (x_min,y_min)和(x_max,y_max)用于表征轮廓所在区域,其中x有关的为行索引，y有关的表征列索引
__global__  void GetCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max, Parameter devpar)
{
	//八零域方向数组，用于更新轮廓点,初始化方向为正右方（0号位），顺时针旋转45°（索引加1）
	const  int direction_y[8] = { 1,1,0,-1,-1,-1,0,1 };
	const  int direction_x[8] = { 0,1,1,1,0,-1,-1,-1 };
	//获取行列索引号
	const int y = (blockIdx.x*blockDim.x + threadIdx.x) * 8;//y代表列数
	const int x = blockIdx.y * 8;//x代表行数
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	/*初始化输出结果值*/
	c_length[Id] = 0;
	x_min[Id] = 0;
	x_max[Id] = 0;
	y_min[Id] = 0;
	y_max[Id] = 0;
	/*初始化轮廓区域信息*/
	short x_pos_max = 0, x_pos_min = 0;
	short y_pos_max = 0, y_pos_min = 0;//保存轮廓所在区域位置信息
	short  Point_counts = 0;//轮廓计数

	/*循环提取轮廓周长信息*/
	if ((y / 8) <= (devpar.ImgWidth / 8))  //qwt815
	{
		for (int i = x; i < (x + 8); i++)
		{
			for (int j = y; j < (y + 8); j++)
			{
				if (255 == src[j + i * devpar.ImgMakeborderWidth])
				{
					//更新边界坐标
					y_pos_max = j;
					y_pos_min = j;
					x_pos_max = i;
					x_pos_min = i;
					Point_counts = 1;//轮廓数目计数值
					// 起始点及当前点  
					short x_pos = i;//行索引
					short y_pos = j;//列索引
					bool first_time = false;//是否时第一次获取轮廓点
					short counts = 0;//用于循环计数
					short curr_d = 0;//方向数组索引计数，取值0-7表示八零域的8各不用的方位
									 // 进行跟踪  
					for (short cLengthCount = 0; cLengthCount < devpar.LengthMax; cLengthCount++)//这里的循环次数需要用界面设置的周长最大值来确定
					{
						//定义根标记点
						short root_x = x_pos;
						short root_y = y_pos;
						//更新轮廓区域边界索引
						x_pos_max = x_pos_max > x_pos ? x_pos_max : x_pos;
						y_pos_max = y_pos_max > y_pos ? y_pos_max : y_pos;
						x_pos_min = x_pos_min < x_pos ? x_pos_min : x_pos;
						y_pos_min = y_pos_min < y_pos ? y_pos_min : y_pos;
						// 循环八次 :用于获取下一个轮廓点
						for (counts = 0; counts < 8; counts++)
						{
							// 防止索引出界  
							curr_d -= curr_d >= 8 ? 8 : 0;
							curr_d += curr_d < 0 ? 8 : 0;
							//事实上，只需要判断7个领域内的信息(除了第一次之外)，第count=6时刚好循环到上一个轮廓点
							if (first_time && (counts == 6))//qwt 9.04 按照以上说法则第一次不能进入if
							{
								continue;
							}
							//更新标记点root;
							root_x = x_pos + direction_x[curr_d];//更新行索引
							root_y = y_pos + direction_y[curr_d];//更新列索引
							//判断点是否越界，超过图像的索引区域
							if (root_x < 0 || root_x >= devpar.ImgHeight*devpar.PictureNum || root_y < 0 || root_y >= devpar.ImgWidth)
							{
								curr_d++;
								continue;
							}
							//如果存在边缘  
							if (255 == src[root_y + root_x * devpar.ImgMakeborderWidth])
							{
								curr_d -= 2;   //更新当前方向  
								Point_counts++;
								//更新b_pt:跟踪的root点  
								x_pos = root_x;
								y_pos = root_y;
								break;   // 跳出for循环  
							}
							curr_d++;
						}   // end for  。
							//跟踪结束条件异常结束
						if (8 == counts || (x_pos >= (x + 8) && y_pos >= (y + 8)))
						{
							break;
						}
						//正常结束
						if (y_pos == j && x_pos == i)
						{
							//保存轮廓信息
							c_length[Id] = Point_counts;
							x_min[Id] = x_pos_min;
							x_max[Id] = x_pos_max;
							y_min[Id] = y_pos_min;
							y_max[Id] = y_pos_max;
							break;
						}//正常结束if
						//判断
						if (cLengthCount == 0)
						{
							first_time = true;
						}
					}//外围for结束			
				}//判断前景点if结束
				j = y_pos_max > j ? y_pos_max : j;//更新横向搜索步长
			}//第一个for结束
			i = x_pos_max > i ? x_pos_max : i;
		}//第二个for 结束
	}
}//核函数结束

 /*面积重心提取*/
 //由于上面getCounter提取的 x_min和x_max表征列， y_min和y_max表征行。
__global__  void GetInfo(unsigned char* src_gray, short *length, short* x_min, short * y_min, short* x_max, short *y_max, short *xpos, short*ypos, short *area, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short count = 0;//用于面积计数
	int sum_gray = 0;//圆点区域的灰度值之和
	int x_sum = 0;//x灰度值加权和
	int y_sum = 0;//y灰度值加权和
	int mThreshold = devpar.Threshold;//二值化阈值
									  //保存方位盒边界
	short xmm = x_min[Id];
	short xmx = x_max[Id];
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = (ymx - ymm + 3) / 4 * 4;
	unsigned char temp0, temp1, temp2, temp3;//用寄存器暂存图像数据，减小全局内存的访问，提高访存效率
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	if (length[Id] > devpar.LengthMin)
	{
		//循环优化,这种情况会多计算一些区域的值（需要处理一下）
		for (int i = xmm; i <= xmx; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
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

				count += temp0 > 0 ? 1 : 0; //面积计算
				count += temp1 > 0 ? 1 : 0;
				count += temp2 > 0 ? 1 : 0;
				count += temp3 > 0 ? 1 : 0;

				sum_gray += temp0 + temp1 + temp2 + temp3;


				x_sum += i * temp0 + i * temp1 + i * temp2 + i * temp3;
				y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;
			}
		area[Id] = count;
		xpos[Id] = x_sum / sum_gray;
		ypos[Id] = y_sum / sum_gray;
	}
}

//筛选非重复信息的函数,这个核函数要启动失败
__global__  void GetTrueInfo(short *xcenter, short *ycenter, short*index, short *sArea, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short temp = 0;
	index[Id] = 0;//索引号清零
	if ((Id > devpar.ColThreadNum) && (Id < devpar.ColThreadNum*(devpar.RowThreadNum - 1)))
	{
		if (xcenter[Id] != 0)
		{
			//判断一个线程获取的坐标是否和与它相邻的右方线程（列+1）、下方线程（行+1）获取的坐标一致。若不一致则输出索引值
			//右
			temp += ((xcenter[Id] == xcenter[Id + 1]) && (ycenter[Id] == ycenter[Id + 1])) ? 1 : 0;//右
			temp += ((xcenter[Id] == xcenter[Id + devpar.ColThreadNum]) && (ycenter[Id] == ycenter[Id + devpar.ColThreadNum])) ? 1 : 0;//下
			temp += ((xcenter[Id] == xcenter[Id - devpar.ColThreadNum + 1]) && (ycenter[Id] == ycenter[Id - devpar.ColThreadNum + 1])) ? 1 : 0;//右上
			temp += ((sArea[Id] > devpar.AreaMin) && (sArea[Id] < devpar.AreaMax)) ? 0 : 1;//qwt8-8
			index[Id] = temp > 0 ? 0 : Id;
		}
	}
}

/*矩形模式的特征提取*/
//输入： 方位盒    灰度图    轮廓图
//输出： 周长  面积  重心坐标
__global__	void GetRecInfo(RecData* mRec, unsigned char *psrcgray, unsigned char *psrccounter,
	short *length, short* area, short *xpos, short *ypos, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x;
	int mThreshold = devpar.Threshold;//二值化阈值
	short count = 0;//用于面积计数
	int clengthCount = 0;
	short clength = 0;//周长计数
	int sum_gray = 0;//圆点区域的灰度值之和
	int x_sum = 0;//x灰度值加权和
	int y_sum = 0;//y灰度值加权和

				  //读取方位盒
	short xmm = mRec[Id].RecXmin;
	short xmx = mRec[Id].RecXmax;
	short ymm = mRec[Id].RecYmin;
	short ymx = mRec[Id].RecYmax;
	short jcount = (ymx - ymm + 3) / 4 * 4;//qwt
	unsigned char temp0, temp1, temp2, temp3;//temp用于计算重心、面积
	unsigned char t0, t1, t2, t3;//t用于计算周长
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	length[Id] = 0;
	//循环优化,这种情况会多计算一些区域的值（需要处理一下）
	for (int i = xmm; i <= xmx; i++)
		for (int j = ymm; j <= ymm + jcount; j = j + 4)
		{
			//防止越界
			temp0 = j    > ymx ? 0 : 1;
			temp1 = j + 1> ymx ? 0 : 1;
			temp2 = j + 2> ymx ? 0 : 1;
			temp3 = j + 3> ymx ? 0 : 1;

			t0 = temp0;//qwt
			t1 = temp1;
			t2 = temp2;
			t3 = temp3;

			//根据二值化阈值
			temp0 *= psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
			temp1 *= psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
			temp2 *= psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
			temp3 *= psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;

			t0 *= psrccounter[j   *t0 + i * devpar.ImgMakeborderWidth];
			t1 *= psrccounter[(j + 1)*t1 + i * devpar.ImgMakeborderWidth];
			t2 *= psrccounter[(j + 2)*t2 + i * devpar.ImgMakeborderWidth];
			t3 *= psrccounter[(j + 3)*t3 + i * devpar.ImgMakeborderWidth];


			count += temp0 > 0 ? 1 : 0; //面积计算
			count += temp1 > 0 ? 1 : 0;
			count += temp2 > 0 ? 1 : 0;
			count += temp3 > 0 ? 1 : 0;


			clengthCount += t0 + t1 + t2 + t3;//周长计算
			sum_gray += temp0 + temp1 + temp2 + temp3;


			x_sum += i * temp0 + i * temp1 + i * temp2 + i * temp3;
			y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;
		}
	//筛选特征
	clength = clengthCount / 255;
	length[Id] = clength;
	area[Id] = count;
	xpos[Id] = x_sum / sum_gray;
	ypos[Id] = y_sum / sum_gray;
}

//-------------------------------------------------------结束----------------------------------------//
void GPUDeviceCheck()
{
	cudaError_t cudaGetDeviceCount(int* count);
	cudaGetDeviceCount(&gDeviceCount);
	for (int i = 0; i<gDeviceCount; i++)
	{
		cudaDeviceProp DevProp;
		cudaGetDeviceProperties(&DevProp, i);
		printf("Device %d has compute capability %d.%d \n", i, DevProp.major, DevProp.minor);
	}
}
// 8位灰度BMP格式图像读取
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
		return 1;
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
			return 1;
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
				return 1;
			}
		}
	}
	else return 0;
}

//在主机端提取方位盒---提取方位盒无重复
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

	//方位数组申明
	const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
	//初始化CPU端方位盒数据
	if (gHostRecData.size() != 0)
		gHostRecData.clear();
	//初始化  方位盒更新数据
	gImgXcenter = 0;
	gImgYcenter = 0;
	gXcenterOffset = 0;
	gYcenterOffset = 0;
	//图像空间分配
	unsigned char *ImgHostdata = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum]; //qwt这里程序有BUG
	unsigned char *m_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//二值化图
	unsigned char *n_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//膨胀图
	unsigned char *c_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//轮廓图	
	
	int Picoffset = devpar.ImgHeight * devpar.ImgWidth;//qwt//读取图片
	for (int j = 0; j < devpar.PictureNum; j++)
	{
		RmwRead8BitBmpFile2Img(path,NULL, ImgHostdata + j*Picoffset, &devpar.ImgWidth, &devpar.ImgHeight);//qwt823
	}
	//二值化
	for (int i = 0; i <devpar.ImgHeight*devpar.PictureNum; i++)
	{
		for (int j = 0; j < devpar.ImgWidth; j++)
		{
			m_ptr[j + i * devpar.ImgWidth] = ImgHostdata[j + i * devpar.ImgWidth] > devpar.Threshold ? 255 : 0;
			c_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
			n_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
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

				bool first_t = false;
				bool tra_flag = false;//设置标志位
									  // 存入  
				c_ptr[j + i * devpar.ImgWidth] = 0;    // 用过的点直接给设置为0  

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
						if (counts == 6 && first_t)
						{
							first_t = true;
							continue;
						}

						// 跟踪的过程，应该是个连续的过程，需要不停的更新搜索的root点  
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
						if (cLength<devpar.LengthMax && (cLength >devpar.LengthMin))
						{
							RecData tempRecData;
							if (xmin - 3 < 0)
								tempRecData.RecXmin = 0;
							else
								tempRecData.RecXmin = xmin - 3;
							if (ymin - 3 < 0)
								tempRecData.RecYmin = 0;
							else
								tempRecData.RecYmin = ymin - 3;
							if (xmax + 3 >devpar.ImgHeight*devpar.PictureNum-1)
								tempRecData.RecXmax = devpar.ImgHeight*devpar.PictureNum - 1;
							else
								tempRecData.RecXmax = xmax + 3;
							if (ymax + 3 > devpar.ImgWidth)
								tempRecData.RecYmax = devpar.ImgWidth - 1;
							else
								tempRecData.RecYmax = ymax + 3;
							if (abs((tempRecData.RecYmax - tempRecData.RecYmin) - (tempRecData.RecXmax - tempRecData.RecXmin))<20)
								gHostRecData.push_back(tempRecData);
						}
						break;
					}
				}  // end if  
			}  // end while  
		}
	//初始化 方位盒更新数据
	if (gHostRecData.size() > 0)
	{
		//提取图像标志点重心累加和
		for (int k = 0; k < gHostRecData.size(); k++)
		{
			//单个矩形盒灰度中心计算
			int GraySum = 0;
			int xGraySum = 0;
			int yGraySum = 0;
			for (int i = gHostRecData[k].RecXmin + 3; i <= gHostRecData[k].RecXmax - 3; i++)
				for (int j = gHostRecData[k].RecYmin + 3; j <= gHostRecData[k].RecYmax - 3; j++)
				{
					if (ImgHostdata[j + i * devpar.ImgWidth] >(devpar.Threshold))
					{
						xGraySum += i * ImgHostdata[j + i * devpar.ImgWidth];
						yGraySum += j * ImgHostdata[j + i * devpar.ImgWidth];
						GraySum += ImgHostdata[j + i * devpar.ImgWidth];
					}
				}
			gImgXcenter += xGraySum / GraySum;
			gImgYcenter += yGraySum / GraySum;
		}
		//计算加权灰度重心
		gImgXcenter /= gHostRecData.size();
		gImgYcenter /= gHostRecData.size();
		//重心偏移量初始化
		gXcenterOffset = 0;
		gYcenterOffset = 0;
		gRectRealNum = gHostRecData.size();//获取方位盒实际数量值
		//规整方位盒数量，利用后续线程配置
		int rRecNum = (gHostRecData.size() + 127) / 128 * 128;
		gHostRecData.resize(rRecNum, RecData{ 0,0,0,0 });
	}
	//释放内存
	delete[]ImgHostdata;
	delete[]m_ptr;
	delete[]n_ptr;
	delete[]c_ptr;
}

//qwt7.26
class R : public Runnable
{
public:
	Parameter Devpar;//变量传参
	~R()
	{
	}
	void Run()
	{
		//调试项
		cudaError_t  err;
		int img_index = 0;
		int Width;
		int Height;
		char strFilename[100];                                          //【1】定义一个字符数组保存----图片的读取路径 
		char saveFilename[100];                                         //【1】定义一个字符数组保存----图片的存储路径
		char* path = "E:\\project\\Simulation\\output";


		//参数计算
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//填充后的宽度计算
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / 8;
		Devpar.ColThreadNum = (Devpar.ImgWidth / 8 + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);

		//设置GPU设备号
		/*主机端*/
		//输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*设备端*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//申请的临时变量空间，包括有方位盒、输出特征的GPU端内存和GPU显存
		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//输出周长
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//面积
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//特征索引号
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//方位盒 xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//设备端输出	周长
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				面积
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				索引号
		}

		while ((img_index + CUDAStreams) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rDev_in[i], gHostImage[img_index + i], sizeof(uchar)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				CopyMakeBorder << <mGrid1, 128, 0, rcS[i] >> > (rDev_in[i], rDev_padding[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				Binarization << <mGrid1, 128, 0, rcS[i] >> > (rDev_padding[i], rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//边界提取
				Dilation << <mGrid1, 128, 0, rcS[i] >> >(rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
			cudaMemcpyAsync(rgpu_2val[i], rgpu_counter[i], sizeof(uchar)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice, rcS[i]);
			}
		
			for (int i = 0; i < CUDAStreams; i++)
			{
				Erosion << <mGrid1, 128, 0, rcS[i] >> > (rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取轮廓和边缘盒
				GetCounter << <mGrid2, 128, 0, rcS[i] >> > (rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], Devpar);//提取轮廓的函数
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取面积和重心//提取特征信息核函数
				GetInfo << <mGrid2, 128, 0, rcS[i] >> > (rDev_padding[i],  gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				GetTrueInfo << <mGrid2, 128, 0, rcS[i] >> > (gpDevXpos[i], gpDevYpos[i], gpDevIndex[i], gpDevArea[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				err=cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			printf("%s\n", cudaGetErrorString(err));
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostIndex[i], gpDevIndex[i], sizeof(short)*	Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaStreamSynchronize(rcS[i]);
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
					sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
					fp = fopen(strFilename, "wb");
					fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
					fclose(fp);
				}
			}
			img_index += gHostImgblock;
		}
		for (int i = 0; i < CUDAStreams; i++)
		{
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
		}
	}
};

//矩形模式的类R
class RecR : public Runnable
{
public:
	Parameter Devpar;//变量传参	
	~RecR()
	{
	}
	void Run()
	{
		cudaError_t err;
		int xCenterSum = 0;//更新方位盒所需数据：
		int yCenterSum = 0;//更新方位盒所需数据
		int img_index = 0;
		char strFilename[100];                                          //【1】定义一个字符数组保存----图片的读取路径 
		char saveFilename[100];                                         //【1】定义一个字符数组保存----图片的存储路径
		char* path = "E:\\project\\Simulation\\output";
		//参数计算
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//填充后的宽度计算
		int mRecCount = gHostRecData.size();//获取方位盒数量
		//核函数声明GRid分配；
		int Gridsize = mRecCount / 128;
		if (Gridsize == 0)//qwt823
			Gridsize = 1;
		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Gridsize, 1, 1);

		/*主机端*/
		//输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		//输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		//拷贝方位盒数据
		RecData *gpRDevRecData[CUDAStreams];//qwt821
		if (gHostRecData.size() > 0) 
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				checkCudaErrors(cudaMalloc((void**)&gpRDevRecData[i], mRecCount * sizeof(RecData)));//
				cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], mRecCount * sizeof(RecData), cudaMemcpyHostToDevice);
			}
		}
		//存储空间分配
		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], mRecCount * sizeof(short), cudaHostAllocDefault));//输出周长
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], mRecCount * sizeof(short), cudaHostAllocDefault));//面积
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], mRecCount * sizeof(short), cudaHostAllocDefault));//重心坐标x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], mRecCount * sizeof(short), cudaHostAllocDefault));//重心坐标y
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], mRecCount * sizeof(short)));//设备端输出	周长
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], mRecCount * sizeof(short)));//				面积
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], mRecCount * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], mRecCount * sizeof(short)));//				ypos
		}
		while ((img_index + CUDAStreams ) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rDev_in[i], gHostImage[img_index + i ], sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				CopyMakeBorder << <mGrid1, 128, 0, rcS[i] >> > (rDev_in[i], rDev_padding[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				Binarization << <mGrid1, 128, 0, rcS[i] >> > (rDev_padding[i],  rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//边界提取
				Dilation << <mGrid1, 128, 0, rcS[i] >> >(rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rgpu_2val[i], rgpu_counter[i], sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				Erosion << <mGrid1, 128, 0, rcS[i] >> > (rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//不同流中的核函数用同一GPU数据时，是否会影响核函数的性能qwt
				GetRecInfo << <mGrid2, 128, 0, rcS[i] >> >(gpRDevRecData[i], rDev_padding[i], rgpu_counter[i],
					gpDevLength[i], gpDevArea[i], gpDevXpos[i], gpDevYpos[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)*   mRecCount, cudaMemcpyDeviceToHost, rcS[i]);
			}
			//printf(" %s\n", cudaGetErrorString(err));//调试项
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)*   mRecCount, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(short)*  mRecCount, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				err=cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(short)*  mRecCount, cudaMemcpyDeviceToHost, rcS[i]);
			}
		//	printf(" %s\n", cudaGetErrorString(err));//调试项
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaStreamSynchronize(rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选打印提取的特征
				vector<CircleInfo>myInfo;
				xCenterSum = 0;
				yCenterSum = 0;
				for (int j = 0; j < mRecCount; j++)
				{
					if (gpHostXpos[i][j] > 0) 
					{
						CircleInfo temp;
						temp.index = j;
						temp.length = gpHostLength[i][j];
						temp.area = gpHostArea[i][j];
						temp.xpos = gpHostXpos[i][j];
						temp.ypos = gpHostYpos[i][j];
						xCenterSum += gpHostXpos[i][j];//重心累加：x
						yCenterSum += gpHostYpos[i][j];//重心累加：y
						myInfo.push_back(temp);
					}
				}
				//包围盒信息更新
				if (myInfo.size() > 0) 
				{
					gXcenterOffset += (gImgXcenter - xCenterSum / myInfo.size());
					gYcenterOffset += (gImgYcenter - yCenterSum / myInfo.size());
					gImgXcenter = xCenterSum / myInfo.size();
					gImgYcenter = yCenterSum / myInfo.size();
				}
				SignPoint.PointNumbers = myInfo.size();
				//输出标志点数据
				if (myInfo.size() > 0)
				{
					FILE* fp;
					sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
					fp = fopen(strFilename, "wb");
					fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
					fclose(fp);
				}
			}
			img_index += gHostImgblock;
			//更新包围盒
			if (img_index % 100 == 0)
			{
				for (int i = 0; i < mRecCount; i++)
				{
					if (gHostRecData[i].RecXmin != 0 && gHostRecData[i].RecYmin != 0)
					{
						gHostRecData[i].RecXmin += gXcenterOffset;//CPU端方位盒
						gHostRecData[i].RecXmax += gXcenterOffset;
						gHostRecData[i].RecYmin += gYcenterOffset;
						gHostRecData[i].RecYmax += gYcenterOffset;
					}
					//边界判断
					if (gHostRecData[i].RecXmin<0 || gHostRecData[i].RecXmin>Devpar.ImgHeight* Devpar.PictureNum||
						gHostRecData[i].RecXmax<0 || gHostRecData[i].RecXmax>Devpar.ImgHeight* Devpar.PictureNum ||
						gHostRecData[i].RecYmin<0 || gHostRecData[i].RecYmin>Devpar.ImgMakeborderWidth ||
						gHostRecData[i].RecYmax<0 || gHostRecData[i].RecYmax>Devpar.ImgMakeborderWidth)
					{
						gHostRecData[i].RecXmin = 0;
						gHostRecData[i].RecXmax = 0;
						gHostRecData[i].RecYmin = 0;
						gHostRecData[i].RecYmax = 0;
					}
				}
				gXcenterOffset = 0;
				gYcenterOffset = 0;
				for (int i = 0; i < CUDAStreams; i++)
				{
					cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], mRecCount * sizeof(RecData), cudaMemcpyHostToDevice);
				}
			}
		}
		for (int i = 0; i < CUDAStreams; i++)
		{
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
		}
	}
};
//矩形模式的类S

//测试原图仿真测试
IMGSIMULATION_API bool SimulationImageTest(const char *path, Infomation *Info) 
{
	cudaError_t  err;
	int mWidth, mHeight;
	gHostPathImgNumber = 20;//测试图片复制数量
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		err = cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth *gStructVarible.PictureNum* sizeof(unsigned char), cudaHostAllocDefault);
	}
	int Picoffset = gStructVarible.ImgHeight * gStructVarible.ImgWidth;//单张图片地址偏移量
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		for (int j = 0; j < gStructVarible.PictureNum; j++)
		{
			RmwRead8BitBmpFile2Img(path, NULL, gHostImage[i]+j*Picoffset, &mWidth, &mHeight);
		}
	}
	//测试图片是否读取成功------------------------------------------------------------------------------------------------------------------
	//cv::Mat img(gStructVarible.ImgHeight*gStructVarible.PictureNum, gStructVarible.ImgWidth, CV_8UC1);
	//for (int i = 0; i < gStructVarible.ImgHeight*gStructVarible.PictureNum; i++)
	//{
	//	uchar* data = img.ptr<uchar>(i);  //获取第i行的首地址。
	//	for (int j = 0; j < gStructVarible.ImgWidth; j++)   //列循环
	//	{
	//		data[j] = gHostImage[10][j + i *   gStructVarible.ImgWidth];
	//	}
	//}
	//imwrite("pic.bmp", img);
	//-------------------------------------------------------------------------------------------------------------------------------------
	if(gStructVarible.RecModelFlag == true)
		 GetImgBoxHost(path);//提取包围盒
	cout << gHostRecData.size();
	/****  单提点测试****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	pExecutor->Init(1, ExtractPointThreads, 1);
	R r;
	RecR recr;

	if (gStructVarible.RecModelFlag == false)//全图模式
	{
		//结构体赋值
		r.Devpar.ImgHeight = gStructVarible.ImgHeight;
		r.Devpar.ImgWidth = gStructVarible.ImgWidth;
		r.Devpar.Threshold = gStructVarible.Threshold;
		r.Devpar.LengthMin = gStructVarible.LengthMin;
		r.Devpar.LengthMax = gStructVarible.LengthMax;
		r.Devpar.AreaMin = gStructVarible.AreaMin;
		r.Devpar.AreaMax = gStructVarible.AreaMax;
		r.Devpar.PictureNum = gStructVarible.PictureNum;
		pExecutor->Execute(&r, 0x01);

		pExecutor->Terminate();
		delete pExecutor;
		
	}
	else //矩形模式
	{
		//结构体赋值
		recr.Devpar.ImgHeight = gStructVarible.ImgHeight;
		recr.Devpar.ImgWidth = gStructVarible.ImgWidth;
		recr.Devpar.Threshold = gStructVarible.Threshold;
		recr.Devpar.LengthMin = gStructVarible.LengthMin;
		recr.Devpar.LengthMax = gStructVarible.LengthMax;
		recr.Devpar.AreaMin = gStructVarible.AreaMin;
		recr.Devpar.AreaMax = gStructVarible.AreaMax;
		recr.Devpar.PictureNum = gStructVarible.PictureNum;
		
		pExecutor->Execute(&recr, 0x01);
		pExecutor->Terminate();
		delete pExecutor;
		
	}
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		err = cudaFreeHost(gHostImage[i]);
		if (gStructVarible.ImgBitDeep == 24)
		{
			delete(gHostColorImage[i]);
		}
		if (err != cudaSuccess)
		{
			return false;
		}
	}
	if (gStructVarible.RecModelFlag == false)//全图模式
	{
		return false;
	}
	else if(gStructVarible.RecModelFlag == true)//矩形模式 
	{
		return true;
	}
}


//全局内存申请
IMGSIMULATION_API void Memory_application(Parameter Devpar)
{
	int paddingWidth = (Devpar.ImgWidth  + 127) / 128 * 128;  //qwt7.26
	cudaSetDevice(0);
	rcS = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));
	for (int i = 0; i < CUDAStreams; i++)
	{
		cudaSetDevice(0);
		cudaStreamCreate(&(rcS[i]));
		cudaMalloc((void**)&rDev_in[i],		 Devpar.ImgHeight *Devpar.ImgWidth  *Devpar.PictureNum* sizeof(unsigned char));
		cudaMalloc((void**)&rDev_padding[i], Devpar.ImgHeight *paddingWidth		*Devpar.PictureNum* sizeof(unsigned char));  //qwt7.26
		cudaMalloc((void**)&rgpu_2val[i],	 Devpar.ImgHeight *paddingWidth		*Devpar.PictureNum* sizeof(unsigned char));
		cudaMalloc((void**)&rgpu_counter[i], Devpar.ImgHeight *paddingWidth		*Devpar.PictureNum* sizeof(unsigned char));
	}
}
//全局内存释放
IMGSIMULATION_API void Memory_release()
{
	for (int i = 0; i<CUDAStreams; i++)
	{
		cudaSetDevice(0);
		//cudaFreeHost(rhost_in[i]);
		cudaFree(rDev_in[i]);
		cudaFree(rDev_padding[i]); //qwt7.26
		cudaFree(rgpu_2val[i]);
		cudaFree(rgpu_counter[i]);
		checkCudaErrors(cudaStreamDestroy(rcS[i]));
	}
}

int main()
{
	//参数设置
	gStructVarible.ImgReadPath = "E:\\project\\Simulation\\data_gray_img\\3.bmp";
	//gStructVarible.ImgSavePath = "E:\\project\\Simulation\\pic_output";
	gStructVarible.DataReadPath = "E:\\project\\Simulation\\output";
	gStructVarible.ImgHeight =5120;
	gStructVarible.ImgWidth = 5120;
	gStructVarible.Threshold =60;
	gStructVarible.LengthMin = 100;
	gStructVarible.LengthMax = 250;
	gStructVarible.AreaMin = 1;
	gStructVarible.AreaMax = 99999;
	gStructVarible.ImgBitDeep = 8;
	gStructVarible.PictureNum = 2;
	gStructVarible.RecModelFlag = false;
	
	////申请内存空间
	Memory_application(gStructVarible);
	Infomation *result = new Infomation;
	SimulationImageTest(gStructVarible.ImgReadPath, result);
	delete result;
	//测试
	char *strfilename = "E:\\project\\Simulation\\output\\1.bin";
	FILE *fr;
	fr = fopen(strfilename, "rb");
	if (fr == NULL)
	{
		cout << "FILE fail open" << endl;
		return 0;
	}
	fseek(fr, 0, SEEK_END);
	long lSize = ftell(fr);
	rewind(fr);
	int num9 = lSize / sizeof(CircleInfo);
	CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*num9);
	fread(RInfo, sizeof(CircleInfo), num9, fr);
	fclose(fr);
	//绘制原点图
	cv::Mat img = cv::imread(gStructVarible.ImgReadPath, cv::IMREAD_COLOR);
	cv::Vec3b pflag(0, 0, 255);
	for (int i = 0; i < num9/gStructVarible.PictureNum; i++)
	{

		img.at<cv::Vec3b>(RInfo[i].xpos%gStructVarible.ImgHeight, RInfo[i].ypos%gStructVarible.ImgWidth) = pflag;
	}

	//绘制包围盒
	/*for (int i = 0; i < gRectRealNum; i++)
	{
		cv::Point  Rmin(gHostRecData[i].RecYmin + 2, gHostRecData[i].RecXmin + 2);
		cv::Point  Rmax(gHostRecData[i].RecYmax - 2, gHostRecData[i].RecXmax - 2);
		rectangle(img, Rmin, Rmax, cv::Scalar(0, 0, 255));
	}*/
	return 0;
}

/*图像生成主函数*/
//int main() 
//{
//	Mat img = imread("E:\\project\\Simulation\\data_gray_img\\4M.bmp",0);
//	Mat img1(2048, 2000, CV_8UC1);
//	for (int i = 0; i < img1.rows; i++)
//	{
//		uchar* data = img.ptr<uchar>(i);  //获取第i行的首地址。
//		uchar* data1 = img1.ptr<uchar>(i);
//		for (int j = 0; j < img1.cols; j++)   //列循环
//			{
//			if(i<(img1.rows-50)&&j<(img1.cols-50))
//				data1[j] = data[j];
//			else data1[j] = 0;
//		}
//		}
//	imwrite("E:\\project\\Simulation\\data_gray_img\\12.bmp", img1);
//	return 0;
//
//}



/*单张普通模式*/
//int main() 
//{
//	cudaError_t err;
//	char *path = "E:\\project\\Simulation\\data_gray_img\\2.bmp";
//	const char * strfilename = "Sig1.bin";
//	Parameter Devpar;
//	Devpar.ImgHeight = 2400;
//	Devpar.ImgWidth = 1600;
//	Devpar.Threshold = 128;
//	Devpar.LengthMin = 30;
//	Devpar.LengthMax = 250;
//	Devpar.AreaMin = 1;
//	Devpar.AreaMax = 99999;
//	Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;
//	Devpar.ColThreadNum = (Devpar.ImgMakeborderWidth / 8 + 127) / 128 * 128;
//	Devpar.RowThreadNum = Devpar.ImgHeight / 8;
//	Devpar.PictureNum = 1;
//	// 线程配置定义
//	dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight, 1);
//	dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);
//	unsigned char *tHostImage;
//	cudaHostAlloc((void**)&tHostImage, Devpar.ImgHeight *  Devpar.ImgWidth * sizeof(unsigned char), cudaHostAllocDefault);
//	RmwRead8BitBmpFile2Img(path,NULL,tHostImage, &Devpar.ImgWidth, &Devpar.ImgHeight);
//	
//	//测试读入图片是否成功------------------------------------------------------------------------------------------------------------
//	cv::Mat img1(Devpar.ImgHeight, Devpar.ImgWidth, CV_8UC1);
//	for (int i = 0; i < Devpar.ImgHeight; i++)
//	{
//		uchar* data = img1.ptr<uchar>(i);  //获取第i行的首地址。
//		for (int j = 0; j < Devpar.ImgWidth; j++)   //列循环
//		{
//			data[j] = tHostImage[j + i * Devpar.ImgWidth];
//		}
//	}
//	//-------------------------------------------------------------------------------------------------------------------------------
//	unsigned char * tDevImage;
//	unsigned char * tDevpad;
//	unsigned char * tDev2val;
//	unsigned char * tDevcounter;
//	cudaMalloc((void**)&tDevImage, sizeof(unsigned char)* Devpar.ImgWidth* Devpar.ImgHeight);
//	cudaMalloc((void**)&tDevpad, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight);
//	cudaMalloc((void**)&tDev2val, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight);
//	cudaMalloc((void**)&tDevcounter, sizeof(unsigned char)* Devpar.ImgMakeborderWidth* Devpar.ImgHeight);
//	//设备端显存申请
//	short *  tDevRecXLeft;
//	short *  tDevRecYLeft;
//	short *  tDevRecXRight;
//	short *  tDevRecYRight;
//	short  *tDevLength;
//	short  *tDevArea;
//	short  *tDevXpos;
//	short  *tDevYpos;
//	short  *tDevIndex;
//	cudaMalloc((void**)&tDevRecXLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//方位盒 xmin
//	cudaMalloc((void**)&tDevRecYLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
//	cudaMalloc((void**)&tDevRecXRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
//	cudaMalloc((void**)&tDevRecYRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
//	cudaMalloc((void**)&tDevLength, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//设备端输出	周长
//	cudaMalloc((void**)&tDevArea, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				面积
//	cudaMalloc((void**)&tDevXpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				xpos
//	cudaMalloc((void**)&tDevYpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				ypos
//	cudaMalloc((void**)&tDevIndex, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				索引号
//																							//输出空间申请
//	short *  tHostRecXLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecYLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecXRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecYRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostLength = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostArea = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostXpos = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostYpos = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostIndex = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	//核函数执行
//	cudaMemcpy(tDevImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth, cudaMemcpyHostToDevice);
//	//执行灰度化，二值化核函数程序
//	CopyMakeBorder << <mGrid1, 128 >> > (tDevImage, tDevpad, Devpar);
//	//执行灰度化，二值化核函数程序
//	Binarization << <mGrid1, 128 >> > (tDevpad, tDev2val, tDevcounter, Devpar);
//	//边界提取
//	Dilation << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
//	cudaMemcpy(tDev2val, tDevcounter, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth, cudaMemcpyDeviceToDevice);
//	Erosion << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
//	//提取周长和包围盒
//	GetCounter << <mGrid2, 128 >> > (tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, Devpar);//提取轮廓的函数																													//测试图像预处理是否成功
//	GetInfo << <mGrid2, 128 >> > (tDevpad, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevXpos, tDevYpos, tDevArea, Devpar);
//	GetTrueInfo << <mGrid2, 128 >> > (tDevXpos, tDevYpos, tDevIndex, tDevArea, Devpar);
//	//拷贝输出结果
//	cudaMemcpy(tHostLength, tDevLength, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostArea, tDevArea, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostXpos, tDevXpos, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostYpos, tDevYpos, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostIndex, tDevIndex, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostRecXLeft, tDevRecXLeft, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostRecYLeft, tDevRecYLeft, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	cudaMemcpy(tHostRecXRight, tDevRecXRight, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	err = cudaMemcpy(tHostRecYRight, tDevRecYRight, sizeof(short)* Devpar.ColThreadNum * Devpar.RowThreadNum, cudaMemcpyDeviceToHost);
//	printf("%s", cudaGetErrorString(err));
//	//输出轮廓图-----------------------------------------------------------------------------------------------------------------------
//	uchar* Src_counter = new uchar[Devpar.ImgHeight*Devpar.ImgMakeborderWidth];
//	cv::Mat img_counter(Devpar.ImgHeight, Devpar.ImgMakeborderWidth, CV_8UC1);
//	err = cudaMemcpy(Src_counter, tDevcounter, sizeof(unsigned char)*Devpar.ImgHeight * Devpar.ImgMakeborderWidth, cudaMemcpyDeviceToHost);
//	printf("%s", cudaGetErrorString(err));
//	for (int i = 0; i < Devpar.ImgHeight; i++)
//	{
//		uchar* data = img_counter.ptr<uchar>(i);  //获取第i行的首地址。
//		for (int j = 0; j < Devpar.ImgMakeborderWidth; j++)   //列循环
//		{
//			data[j] = Src_counter[j + Devpar.ImgMakeborderWidth* i];
//		}
//	}
//	//将灰度图转换为彩图 ,包围盒和圆心坐标都将绘制在img_out_rec上面
//	cv::Mat img_out(Devpar.ImgHeight, Devpar.ImgMakeborderWidth, CV_8UC3);
//	cv::cvtColor(img_counter, img_out, cv::COLOR_GRAY2BGR);
//	//----------------------------------------------------------------------------------------------------------------------------------
//	//筛选结果
//	vector<CircleInfo>myInfo;
//	for (int j = 0; j < Devpar.ColThreadNum * Devpar.RowThreadNum; j++)
//	{
//		if (tHostIndex[j] != 0)
//		{
//			CircleInfo temp;
//			temp.index = (short)j;
//			temp.length = tHostLength[j];
//			temp.area = tHostArea[j];
//			temp.xpos = tHostXpos[j];
//			temp.ypos = tHostYpos[j];
//			myInfo.push_back(temp);
//			//绘制方位盒
//			cv::Point  Rmin(tHostRecYLeft[j] - 1, tHostRecXLeft[j] - 1);
//			cv::Point  Rmax(tHostRecYRight[j] + 1, tHostRecXRight[j] + 1);
//			cv::rectangle(img_out, Rmin, Rmax, cv::Scalar(0, 0, 255));
//			img_out.at<cv::Vec3b>(temp.xpos, temp.ypos) = cv::Vec3b(0, 0, 255);
//		}
//	}
//	//写出特征
//	if (myInfo.size() > 0)
//	{
//		FILE* fp;
//		fp = fopen(strfilename, "wb");
//		fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
//		fclose(fp);
//	}
//	//读取磁盘特征，并绘制圆心坐标
//	FILE *fr;
//	fr = fopen(strfilename, "rb");
//	if (fr != NULL)
//	{
//		fseek(fr, 0, SEEK_END);
//		long lSize = ftell(fr);
//		rewind(fr);
//		int num9 = lSize / sizeof(CircleInfo);
//		CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*num9);
//		fread(RInfo, sizeof(CircleInfo), num9, fr);
//		//绘制原点图
//		for (int i = 0; i < num9; i++)
//		{
//			img_out.at<cv::Vec3b>(RInfo[i].xpos, RInfo[i].ypos) = cv::Vec3b(0, 255, 0);
//		}
//	}
//	fclose(fr);
//	//释放内存
//	cudaFreeHost(tHostImage);
//	cudaFree(tDevRecXLeft);
//	cudaFree(tDevRecYLeft);
//	cudaFree(tDevRecXRight);
//	cudaFree(tDevRecYRight);
//	cudaFree(tDevLength);
//	cudaFree(tDevArea);
//	cudaFree(tDevXpos);
//	cudaFree(tDevYpos);
//	cudaFree(tDevIndex);
//	cudaFree(tDevImage);
//	cudaFree(tDevpad);
//	cudaFree(tDev2val);
//	cudaFree(tDevcounter);
//	delete[]tHostRecXLeft;
//	delete[]tHostRecYLeft;
//	delete[]tHostRecXRight;
//	delete[] tHostRecYRight;
//	delete[]tHostLength;
//	delete[]tHostArea;
//	delete[]tHostXpos;
//	delete[]tHostYpos;
//	delete[]tHostIndex;
//	delete[]Src_counter;
//	return 0;
//}

