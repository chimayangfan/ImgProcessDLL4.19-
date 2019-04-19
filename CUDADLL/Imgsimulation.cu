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
#pragma comment( lib, "GdiPlus.lib" )
using namespace Gdiplus;
using namespace std;
using namespace cv;

//根据设备性能定义
#define CPUThreads 2
#define CUDAStreams 5
int gHostImgblock = CPUThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
//根据图片大小定义block和thread个数 
#define gThreshold 60   //阈值化的阈值
#define counterNum 640  //这个是定义的提取轮廓信息时所用线程配置
#define gLengthMax 300//周长的最大值
#define gLengthMin 30 //周长的最小值 
const int gImgHeight = 5120;//列数
const int gImgWidth = 5120; //行数
const int gThreadNum = gImgHeight * gImgWidth / 64;

#define Pretreatment
#ifdef Pretreatment
#define ReadImageNumber 250
unsigned char* gHostImage[ReadImageNumber];
#endif // Pretreatment

unsigned char* rhost_in[CUDAStreams];//页锁定内存
unsigned char* rDev_in[CUDAStreams];//设备内存
unsigned char* rgpu_2val[CUDAStreams];//二值化图
unsigned char* rgpu_counter[CUDAStreams];//轮廓图，在执行findcountores之后才生成

unsigned char* shost_in[CUDAStreams];//页锁定内存
unsigned char* sDev_in[CUDAStreams];//设备内存
unsigned char* sgpu_2val[CUDAStreams];//二值化图
unsigned char* sgpu_counter[CUDAStreams];//轮廓图，在执行findcountores之后才生成

dim3 mGrid(20, 5120);//这个参数得设为全局变量
dim3 mGrid2(5, 640);

cudaStream_t *rcS;
cudaStream_t *scS;

/*灰度化和二值化*/
//该核函数线程配置位  <<<(5,5120),1024>>>  【我的电脑block中最多1024, grid中一行block处理图像的一行】;   src为原图像   dst为二值化图像  dst2为灰度图
__global__ void Graybmp(unsigned char *src_gray, unsigned char *dst_2val, unsigned char *dst_counter)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//【二维grid和一维block】
	int temp = int(src_gray[Id]);//寄存器保存像素，提高访存效率								
	if (Id < gImgWidth*gImgHeight)
	{
		dst_2val[Id] = unsigned char(255 * int(temp>gThreshold));//二值化，利用计算代替分支结构
		dst_counter[Id] = unsigned char(255 * int(temp>gThreshold));
	}
}

/*获取轮廓（边缘检测）*/
//利用十字型4领域腐蚀对原图像素点逐一操作，线程配置<<<(5,5120),1024>>>
//膨胀-----输入二值化图
__global__  void dilation(unsigned char *src, unsigned char *dst)
{
	const int Id_x = threadIdx.x + blockIdx.x *blockDim.x;//Id_x代表行信息  Id_y代表列信息
	const int Id_y = blockIdx.y;//Id_y代表列信息
	int temp;
	if (Id_x > 1 && Id_x < (gImgWidth - 2) && Id_y>1 && Id_y < gridDim.y - 1)
	{
		if (src[Id_x + Id_y * gImgWidth] == 0)
		{
			temp = int(src[Id_x - 1 + (Id_y - 1)*gImgWidth]) + int(src[Id_x + (Id_y - 1)*gImgWidth]) + int(src[Id_x + 1 + (Id_y - 1)*gImgWidth])
				+ int(src[Id_x - 1 + Id_y * gImgWidth]) + int(src[Id_x + 1 + Id_y * gImgWidth]) +
				int(src[Id_x - 1 + (Id_y + 1)*gImgWidth]) + int(src[Id_x + (Id_y + 1)*gImgWidth]) + int(src[Id_x + 1 + (Id_y + 1)*gImgWidth]);//用4领域膨胀
			dst[Id_x + Id_y * gImgWidth] = temp > 0 ? 255 : 0;
		}
	}

}


//腐蚀
__global__  void erosion(unsigned char *src, unsigned char *dst)
{
	const int Id_x = threadIdx.x + blockIdx.x *blockDim.x;//Id_x代表行信息  Id_y代表列信息
	const int Id_y = blockIdx.y;//Id_y代表列信息
	int temp;
	//利用4领域值掏空内部点，提取轮廓信息，现在的dst就是存储轮廓的信息
	if (Id_x > 0 && Id_x < (gImgWidth - 1) && Id_y>0 && Id_y < gridDim.y)
	{
		if (src[Id_x + Id_y * gImgWidth] != 0)
		{
			temp = int(src[Id_x + (Id_y - 1)*gImgWidth]) + int(src[Id_x - 1 + Id_y * gImgWidth]) +
				int(src[Id_x + 1 + Id_y * gImgWidth]) + int(src[Id_x + (Id_y + 1)*gImgWidth]);//用4领域腐蚀
			dst[Id_x + Id_y * gImgWidth] = temp >= 1020 ? 0 : 255;
		}
	}
}


/*轮廓提取*/
//利用八领域追踪法获取轮廓，线程配置为<<<640，640>>>；   一个线程处理16*16大小的像素区域，提取准则是右下边界优先原则 
//src为轮廓数组（边缘检测结果），c_length提取的周长值，   (x_min,y_min)和(x_max,y_max)用于表征轮廓所在区域,其中x有关的为行索引，y有关的表征列索引
__global__  void getCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max)
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
	for (int i = x; i < (x + 8); i++)
	{
		for (int j = y; j < (y + 8); j++)
		{
			if (255 == src[j + i * gImgWidth])
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
				bool first_time = true;//是否时第一次获取轮廓点
				short counts = 0;//用于循环计数
				short curr_d = 0;//方向数组索引计数，取值0-7表示八零域的8各不用的方位
								 // 进行跟踪  
				for (short cLengthCount = 0; cLengthCount<gLengthMax; cLengthCount++)//这里的循环次数需要用界面设置的周长最大值来确定
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

						//事实上，只需要判断7个领域内的信息(除了第一次之外)
						if (first_time && (counts == 6))
						{
							first_time = false;
							continue;
						}
						//更新标记点root;

						root_x = x_pos + direction_x[curr_d];//更新行索引
						root_y = y_pos + direction_y[curr_d];//更新列索引

															 //判断点是否越界，超过图像的索引区域
						if (root_x < 0 || root_x >= gImgHeight || root_y < 0 || root_y >= gImgWidth)
						{
							curr_d++;
							continue;
						}
						//如果存在边缘  
						if (255 == src[root_y + root_x * gImgWidth])
						{
							curr_d -= 2;   //更新当前方向  
							Point_counts++;
							//更新b_pt:跟踪的root点  
							x_pos = root_x;
							y_pos = root_y;
							break;   // 跳出for循环  
						}
						curr_d++;
					}   // end for  

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
				}//外围for结束			
			}//判断前景点if结束
			j = y_pos_max>j ? y_pos_max : j;//更新横向搜索步长
		}//第一个for结束
		i = x_pos_max>i ? x_pos_max : i;
	}//第二个for 结束
}//核函数结束

 //由于上面getCounter提取的 x_min和x_max表征列， y_min和y_max表征行。
 //这个核函数与getCounter调用的线程数都是<<<640,640>>>；用于返回地址，面积，重心
__global__  void getInfo(unsigned char* src_gray, unsigned char* src_counter, short *length, short* x_min, short * y_min, short* x_max, short *y_max, short *xpos, short*ypos, short *area)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short count = 0;//用于面积计数
	int sum_gray = 0;//圆点区域的灰度值之和
	int x_sum = 0;//x灰度值加权和
	int y_sum = 0;//y灰度值加权和
				  //由于每次循环判断都要访问边界，所以改用寄存器存储边界。
	short xmm = x_min[Id];
	short xmx = x_max[Id];
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = ((ymx - ymm) / 4 + 1) * 4;
	unsigned char temp0, temp1, temp2, temp3;//用寄存器暂存图像数据，减小全局内存的访问，提高访存效率
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	if (length[Id] > gLengthMin)
	{
		//循环优化,这种情况会多计算一些区域的值（需要处理一下）
		for (int i = xmm; i <= xmx; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
				//防止越界
				temp0 = j > ymx ? 0 : 1;
				temp1 = j > ymx ? 0 : 1;
				temp2 = j > ymx ? 0 : 1;
				temp3 = j > ymx ? 0 : 1;


				temp0 *= src_gray[j + i * gImgWidth];
				temp1 *= src_gray[j + 1 + i * gImgWidth];
				temp2 *= src_gray[j + 2 + i * gImgWidth];
				temp3 *= src_gray[j + 3 + i * gImgWidth];

				count += temp0>0 ? 1 : 0; //面积计算
				count += temp1>0 ? 1 : 0;
				count += temp2>0 ? 1 : 0;
				count += temp3>0 ? 1 : 0;

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
__global__  void getTrueInfo(short *xcenter, short *ycenter, short*index)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short temp = 0;
	index[Id] = 0;//索引号清零
	if ((Id > counterNum) && (Id < counterNum*(counterNum - 1)))
	{
		if (xcenter[Id] != 0)
		{
			//判断一个线程获取的坐标是否和与它相邻的右方线程（列+1）、下方线程（行+1）获取的坐标一致。若不一致则输出索引值
			//右
			temp += ((xcenter[Id] == xcenter[Id + 1]) && (ycenter[Id] == ycenter[Id + 1])) ? 1 : 0;//右
			temp += ((xcenter[Id] == xcenter[Id + 640]) && (ycenter[Id] == ycenter[Id + 640])) ? 1 : 0;//下
			temp += ((xcenter[Id] == xcenter[Id - 639]) && (ycenter[Id] == ycenter[Id - 639])) ? 1 : 0;//右上
			index[Id] = temp > 0 ? 0 : Id;
		}
	}
	//新加的区域选取模式
}

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

//8位灰度BMP格式图像读取
unsigned char *RmwRead8BitBmpFile2Img(const char * filename, int *width, int *height) {
	FILE *binFile;
	unsigned char *pImg = NULL;
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER bmpHeader;
	BOOL isRead = TRUE;
	int linenum, ex; //linenum:一行像素的字节总数，包括填充字节 

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
	linenum = (*width * 1 + 3) / 4 * 4;
	ex = linenum - *width * 1;         //每一行的填充字节

	fseek(binFile, fileHeader.bfOffBits, SEEK_SET);
	pImg = new unsigned char[(*width)*(*height)];
	if (pImg != NULL) {
		for (int i = 0; i<*height; i++) {
			int r = fread(pImg + (*height - i - 1)*(*width), sizeof(unsigned char), *width, binFile);
			if (r != *width) {
				delete pImg;
				fclose(binFile);
				return NULL;
			}
			fseek(binFile, ex, SEEK_CUR);
		}
	}
	fclose(binFile);
	return pImg;
}

//BMP格式图像写入
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

class R : public Runnable
{
public:
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
		char* path = "C:\\pic\\img_data";

		//设置GPU设备号
		cudaSetDevice(0);
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

		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//输出周长
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//面积
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//特征索引号
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], gThreadNum * sizeof(short)));//方位盒 xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], gThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], gThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], gThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], gThreadNum * sizeof(short)));//设备端输出	周长
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], gThreadNum * sizeof(short)));//				面积
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], gThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], gThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], gThreadNum * sizeof(short)));//				索引号
		}

		while ((img_index + CUDAStreams) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rDev_in[i], gHostImage[img_index + i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyHostToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				Graybmp << <mGrid, 256, 0, rcS[i] >> > (rDev_in[i], rgpu_2val[i], rgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//边界提取
				dilation << <mGrid, 256, 0, rcS[i] >> >(rgpu_2val[i], rgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rgpu_2val[i], rgpu_counter[i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyDeviceToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				erosion << <mGrid, 256, 0, rcS[i] >> > (rgpu_2val[i], rgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取轮廓和边缘盒
				getCounter << <mGrid2, 128, 0, rcS[i] >> > (rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i]);//提取轮廓的函数
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取面积和重心//提取特征信息核函数
				getInfo << <mGrid2, 128, 0, rcS[i] >> > (rDev_in[i], rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				getTrueInfo << <mGrid2, 128, 0, rcS[i] >> > (gpDevXpos[i], gpDevYpos[i], gpDevIndex[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostIndex[i], gpDevIndex[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
				fp = fopen(strFilename, "wb");
				fwrite(gpHostLength[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostArea[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostXpos[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostYpos[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostIndex[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fclose(fp);
			}
			img_index += gHostImgblock;
		}
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamSynchronize(rcS[i]);
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

class S : public Runnable
{
public:
	~S()
	{
	}
	void Run()
	{
		int img_index = 0;
		int Width;
		int Height;
		char strFilename[100];                                          //【1】定义一个字符数组保存----图片的读取路径 
		char saveFilename[100];                                         //【1】定义一个字符数组保存----图片的存储路径
		char* path = "C:\\pic\\img_data";

		//设置GPU设备号
		cudaSetDevice(1);
		/*主机端*/
		//输入
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*//测试显示用代码，调试成功后屏蔽↓↓↓↓
		//临时变量：主机端包围盒
		short *rec_xmin[CUDAStreams];
		short *rec_ymin[CUDAStreams];
		short *rec_xmax[CUDAStreams];
		short *rec_ymax[CUDAStreams];
		/*测试显示用代码，调试成功后屏蔽↑↑↑↑*/
		/*设备端*/
		short *gpDevRecXLeft[CUDAStreams];
		short *gpDevRecYLeft[CUDAStreams];
		short *gpDevRecXRight[CUDAStreams];
		short *gpDevRecYRight[CUDAStreams];
		//输出
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//输出周长
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//面积
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//重心坐标y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//特征索引号
			/*//测试显示用代码，调试成功后屏蔽↓↓↓↓
			//临时变量：主机端包围盒
			checkCudaErrors(cudaHostAlloc((void**)&rec_xmin[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//xmin
			checkCudaErrors(cudaHostAlloc((void**)&rec_ymin[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//ymin
			checkCudaErrors(cudaHostAlloc((void**)&rec_xmax[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//xmax
			checkCudaErrors(cudaHostAlloc((void**)&rec_ymax[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//ymax
			/*测试显示用代码，调试成功后屏蔽↑↑↑↑*/
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], gThreadNum * sizeof(short)));//方位盒 xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], gThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], gThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], gThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], gThreadNum * sizeof(short)));//设备端输出	周长
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], gThreadNum * sizeof(short)));//				面积
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], gThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], gThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], gThreadNum * sizeof(short)));//				索引号
		}

		while ((img_index + CUDAStreams * 2) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				//cudaMemcpyAsync(sDev_in[i], gHostImage[img_index + i + CUDAStreams], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyHostToDevice, scS[i]);
				cudaMemcpy(sDev_in[i], gHostImage[img_index + i + CUDAStreams], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyHostToDevice);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//执行灰度化，二值化核函数程序
				Graybmp << <mGrid, 256, 0, scS[i] >> > (sDev_in[i], sgpu_2val[i], sgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//边界提取
				dilation << <mGrid, 256, 0, scS[i] >> >(sgpu_2val[i], sgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(sgpu_2val[i], sgpu_counter[i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyDeviceToDevice, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				erosion << <mGrid, 256, 0, scS[i] >> > (sgpu_2val[i], sgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取轮廓和边缘盒
				getCounter << <mGrid2, 128, 0, scS[i] >> > (sgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i]);//提取轮廓的函数
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//提取面积和重心//提取特征信息核函数
				getInfo << <mGrid2, 128, 0, scS[i] >> > (sDev_in[i], sgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//筛选提取出的特征数组的非重复信息
				getTrueInfo << <mGrid2, 128, 0, scS[i] >> > (gpDevXpos[i], gpDevYpos[i], gpDevIndex[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostArea[i], gpDevArea[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostXpos[i], gpDevXpos[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostYpos[i], gpDevYpos[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostIndex[i], gpDevIndex[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			/*//测试显示用代码，调试成功后屏蔽↓↓↓↓
			for (int i = 0; i < CUDAStreams; i++)
			{
			cudaMemcpyAsync(rec_xmin[i], gpDevRecXLeft[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
			cudaMemcpyAsync(rec_ymin[i], gpDevRecXRight[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
			cudaMemcpyAsync(rec_xmax[i], gpDevRecYLeft[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
			cudaMemcpyAsync(rec_ymax[i], gpDevRecYRight[i], sizeof(short)* counterNum * counterNum, cudaMemcpyDeviceToHost, scS[i]);
			}

			for (int i = 0; i < CUDAStreams; i++)
			{
			//测试1：
			uchar *img_counter = new uchar[gImgHeight *gImgWidth];
			cudaMemcpy(img_counter, sgpu_counter[i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyDeviceToHost);

			//printf("%s", cudaGetErrorString(err));
			Mat img_out_counter(gImgWidth, gImgHeight , CV_8UC1);
			for (int j = 0; j < gImgWidth; j++)
			{
			uchar* data = img_out_counter.ptr<uchar>(j);  //获取第i行的首地址。
			for (int k = 0; k < gImgHeight ; k++)   //列循环
			{
			data[k] = img_counter[k + j * gImgHeight ];
			}
			}
			//测试2-标记点是否提取正确对
			Mat img_out_rect(gImgWidth, gImgHeight , CV_8UC3);
			cvtColor(img_out_counter, img_out_rect, COLOR_GRAY2BGR);

			//筛选打印提取的特征
			vector<CircleInfo>myInfo;
			for (int j = 0; j < counterNum * counterNum; j++)
			{
			if (gpHostIndex[i][j] != 0)
			{
			CircleInfo temp;
			temp.xpos = gpHostXpos[i][j];
			temp.ypos = gpHostYpos[i][j];
			temp.cLength = gpHostLength[i][j];
			temp.sArea = gpHostArea[i][j];
			myInfo.push_back(temp);
			//画出包围盒矩形框
			cv::Point temptop(rec_ymin[i][j] - 1, rec_xmin[i][j] - 1);
			cv::Point tempdown(rec_ymax[i][j] + 1, rec_xmax[i][j] + 1);
			rectangle(img_out_rect, temptop, tempdown, Scalar(0, 0, 255), 1, 1, 0);
			img_out_rect.at<Vec3b>(gpHostXpos[i][j], gpHostYpos[i][j])[2] = 255;
			}

			}
			//cout << endl << "特征点数目" << myInfo.size() << endl;

			//sprintf_s(strFilename, "C:\\pic\\img_write\\%d.bmp", img_index + i + CUDAStreams + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
			//imwrite(strFilename, img_out_counter);
			delete[]img_counter;
			}
			/*测试显示用代码，调试成功后屏蔽↑↑↑↑*/
			for (int i = 0; i < CUDAStreams; i++)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + CUDAStreams + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间
				fp = fopen(strFilename, "wb");
				fwrite(gpHostLength[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostArea[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostXpos[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostYpos[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fwrite(gpHostIndex[i], 1, sizeof(short)* counterNum * counterNum, fp);
				fclose(fp);
			}
			img_index += gHostImgblock;
		}
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamSynchronize(scS[i]);
		}
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaFreeHost(gpHostLength[i]);
			cudaFreeHost(gpHostArea[i]);
			cudaFreeHost(gpHostXpos[i]);
			cudaFreeHost(gpHostYpos[i]);
			cudaFreeHost(gpHostIndex[i]);
			/*//测试显示用代码，调试成功后屏蔽↓↓↓↓
			cudaFreeHost(rec_xmin[i]);
			cudaFreeHost(rec_ymin[i]);
			cudaFreeHost(rec_xmax[i]);
			cudaFreeHost(rec_ymax[i]);
			/*测试显示用代码，调试成功后屏蔽↑↑↑↑*/
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
/*对外接口函数*/

//参数设置
//图像路径、格式检查
IMGSIMULATION_API void Image_path_check(const char *path, const char *exten)
{
	Directory dir;
	string filepath(path);
	string fileexten(exten);

	vector<string> filenames = dir.GetListFiles(filepath, fileexten, false);

	if (filenames.size() == NULL)
	{
		perror(" There is no .BMP file! ");
		exit(0);
	}
	else
	{
		gHostPathImgNumber = filenames.size();
	}
	//图像预处理，从硬盘批量读入内存
	#ifdef Pretreatment
		char strFilename[100];
		int mWidth;
		int mHeight;
		for (int i = 0; i < ReadImageNumber; i++)
		{
			sprintf_s(strFilename, "%s\\%d.bmp", path, i + 1); //【3】将图片的路径名动态的写入到strFilename这个地址的内存空间 
			checkCudaErrors(cudaHostAlloc((void**)&gHostImage[i], gImgHeight * gImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
			gHostImage[i] = RmwRead8BitBmpFile2Img(strFilename, &mWidth, &mHeight);
		}
	#endif // Pretreatment
}

//测试原图仿真测试
IMGSIMULATION_API void SimulationImageTest(const char *path, int mWidth, int mHeight)
{
	char strFilename[150];
	for (int i = 0; i < 100; i++)
	{
		checkCudaErrors(cudaHostAlloc((void**)&gHostImage[i], gImgHeight * gImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
		gHostImage[i] = RmwRead8BitBmpFile2Img(path, &mWidth, &mHeight);
	}
	gHostPathImgNumber = 100;
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	pExecutor->Init(1, CPUThreads, 1);
	R r;
	S s;

	pExecutor->Execute(&r, 0x01);
	pExecutor->Execute(&s, 0x02);

	pExecutor->Terminate();
	delete pExecutor;
	//释放内存
	for (int i = 0; i < 100; i++)
	{
		cudaFreeHost(gHostImage[i]);
	}
}

//全局内存申请
IMGSIMULATION_API void Memory_application()
{
	cudaSetDevice(0);
	rcS = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));
	cudaSetDevice(1);
	scS = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));
	for (int i = 0; i < CUDAStreams; i++)
	{
		cudaSetDevice(0);
		checkCudaErrors(cudaStreamCreate(&(rcS[i])));
		//checkCudaErrors(cudaHostAlloc((void**)&rhost_in[i], gImgHeight * gImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc((void**)&rDev_in[i], gImgHeight * gImgWidth * sizeof(unsigned char)));
		cudaMalloc((void**)&rgpu_2val[i], sizeof(unsigned char)*gImgHeight*gImgWidth);
		cudaMalloc((void**)&rgpu_counter[i], sizeof(unsigned char)*gImgHeight*gImgWidth);

		cudaSetDevice(1);
		checkCudaErrors(cudaStreamCreate(&(scS[i])));
		//checkCudaErrors(cudaHostAlloc((void**)&shost_in[i], gImgHeight * gImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc((void**)&sDev_in[i], gImgHeight * gImgWidth * sizeof(unsigned char)));
		cudaMalloc((void**)&sgpu_2val[i], sizeof(unsigned char)*gImgHeight*gImgWidth);
		cudaMalloc((void**)&sgpu_counter[i], sizeof(unsigned char)*gImgHeight*gImgWidth);
	}

}

//全局内存释放
IMGSIMULATION_API void Memory_release()
{
	#ifdef Pretreatment
		for (int i = 0; i < ReadImageNumber; i++)
		{
			cudaFreeHost(gHostImage[i]);
		}
	#endif // Pretreatment
	for (int i = 0; i<CUDAStreams; i++)
	{
		cudaSetDevice(0);
		//cudaFreeHost(rhost_in[i]);
		cudaFree(rDev_in[i]);
		cudaFree(rgpu_2val[i]);
		cudaFree(rgpu_counter[i]);
		checkCudaErrors(cudaStreamDestroy(rcS[i]));

		cudaSetDevice(1);
		//cudaFreeHost(shost_in[i]);
		cudaFree(sDev_in[i]);
		cudaFree(sgpu_2val[i]);
		cudaFree(sgpu_counter[i]);
		checkCudaErrors(cudaStreamDestroy(scS[i]));
	}
}