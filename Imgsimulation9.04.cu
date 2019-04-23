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
#include <helper_cuda.h>//������
#include <Windows.h>
#include <GdiPlus.h>

#include <helper_string.h>
#include <npp.h>

#pragma comment( lib, "GdiPlus.lib" )
using namespace Gdiplus;
using namespace std;
using namespace cv;

//�����豸���ܶ���
#define ExtractPointThreads 1
#define CUDAStreams 2
int gHostImgblock = ExtractPointThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
//����ͼƬ��С����block��thread���� 
Parameter gStructVarible{NULL,NULL,NULL,8,5120,5120,5120,60,30,300,640,640,0,9999,1,false};
Infomation SignPoint;

#define Pretreatment
	#ifdef Pretreatment
	#define ReadImageNumber 250
#endif // Pretreatment
unsigned char* gHostImage[250] = { NULL };
unsigned char* gHostColorImage[250] = { NULL };

unsigned char* rhost_in[CUDAStreams];//ҳ�����ڴ�
unsigned char* rDev_in[CUDAStreams];//�豸�ڴ�
unsigned char* rDev_padding[CUDAStreams];//���߽���ͼ���ڴ�   qwt7.26
unsigned char* rgpu_2val[CUDAStreams];//��ֵ��ͼ
unsigned char* rgpu_counter[CUDAStreams];//����ͼ����ִ��findcountores֮�������

cudaStream_t *rcS;


//-------------------------��λ��Model����-----------------------------//
typedef struct
{
	short RecXmin;
	short RecYmin;
	short RecXmax;
	short RecYmax;
}RecData;//��λ�����ݽṹ
vector<RecData> gHostRecData;//CPU��λ����������
int gRectRealNum;//��λ�е�ʵ������
//��λ�и�������
int gImgXcenter;//ͼƬ�Ҷ����ĵ㣨�����õĻҶ����ļ�Ȩƽ����
int gImgYcenter;
int gXcenterOffset;//��Χ��ƫ����(��Χ�и���ʱ���õ�ֵ)
int gYcenterOffset;
//��������
struct CircleInfo
{
	short index;
	short length;
	short area;
	short xpos;
	short ypos;
};
//-------------------------------------------------------����----------------------------------------//

/*------------------------------------------------�˺���--------------------------------------------------*/
//--------------------------------------------------------��ʼ---------------------------------------------//

/*���ͼ��߽�*/
//����Ϊԭͼͼ��ͼ��߶ȡ�ͼ����  ���Ϊ����Ŀ��  ����Ŀ�ȼ��㹫ʽ   int imgWidth = (width + 127) / 128 * 128;
__global__ void  CopyMakeBorder(const unsigned char *src, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//Id_y��ʾͼ��������
	const int Id_x = blockIdx.y;
	if (Id_y <  devpar.ImgWidth)
	{
		dst[Id_y + Id_x * devpar.ImgMakeborderWidth] = src[Id_y + Id_x * devpar.ImgWidth];
	}
}

/*��ֵ��*/
__global__ void Binarization(unsigned char *psrcgray, unsigned char *pdst2val, unsigned char *pdstcounter, Parameter devpar)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//����άgrid��һάblock��
	int temp = int(psrcgray[Id]);//�Ĵ����������أ���߷ô�Ч��								
	if (Id < devpar.ImgMakeborderWidth * devpar.ImgHeight*devpar.PictureNum)
	{
		pdst2val[Id] = unsigned char(255 * int(temp>devpar.Threshold));//��ֵ�������ü�������֧�ṹ
		pdstcounter[Id] = unsigned char(255 * int(temp>devpar.Threshold));
	}
}

/*��ȡ��������Ե��⣩*/
//����
__global__  void Dilation(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_x��������Ϣ  Id_y��������Ϣ
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

//��ʴ
__global__  void Erosion(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y��������Ϣ  Id_x��������Ϣ
	const int Id_x = blockIdx.y;//Id_x��������Ϣ
	int temp;
	//����4����ֵ�Ϳ��ڲ��㣬��ȡ������Ϣ�����ڵ�dst���Ǵ洢��������Ϣ
	if (Id_y > 0 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x <devpar.ImgHeight*devpar.PictureNum - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] != 0)
		{
			temp = int(psrc[Id_y + (Id_x - 1)*devpar.ImgMakeborderWidth]) + int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)*devpar.ImgMakeborderWidth]);//��4����ʴ
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp >= 1020 ? 0 : 255;
		}
	}
}

/*������ȡ*/
//srcΪ�������飨��Ե���������c_length��ȡ���ܳ�ֵ��   (x_min,y_min)��(x_max,y_max)���ڱ���������������,����x�йص�Ϊ��������y�йصı���������
__global__  void GetCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max, Parameter devpar)
{
	//�����������飬���ڸ���������,��ʼ������Ϊ���ҷ���0��λ����˳ʱ����ת45�㣨������1��
	const  int direction_y[8] = { 1,1,0,-1,-1,-1,0,1 };
	const  int direction_x[8] = { 0,1,1,1,0,-1,-1,-1 };
	//��ȡ����������
	const int y = (blockIdx.x*blockDim.x + threadIdx.x) * 8;//y��������
	const int x = blockIdx.y * 8;//x��������
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	/*��ʼ��������ֵ*/
	c_length[Id] = 0;
	x_min[Id] = 0;
	x_max[Id] = 0;
	y_min[Id] = 0;
	y_max[Id] = 0;
	/*��ʼ������������Ϣ*/
	short x_pos_max = 0, x_pos_min = 0;
	short y_pos_max = 0, y_pos_min = 0;//����������������λ����Ϣ
	short  Point_counts = 0;//��������

	/*ѭ����ȡ�����ܳ���Ϣ*/
	if ((y / 8) <= (devpar.ImgWidth / 8))  //qwt815
	{
		for (int i = x; i < (x + 8); i++)
		{
			for (int j = y; j < (y + 8); j++)
			{
				if (255 == src[j + i * devpar.ImgMakeborderWidth])
				{
					//���±߽�����
					y_pos_max = j;
					y_pos_min = j;
					x_pos_max = i;
					x_pos_min = i;
					Point_counts = 1;//������Ŀ����ֵ
					// ��ʼ�㼰��ǰ��  
					short x_pos = i;//������
					short y_pos = j;//������
					bool first_time = false;//�Ƿ�ʱ��һ�λ�ȡ������
					short counts = 0;//����ѭ������
					short curr_d = 0;//������������������ȡֵ0-7��ʾ�������8�����õķ�λ
									 // ���и���  
					for (short cLengthCount = 0; cLengthCount < devpar.LengthMax; cLengthCount++)//�����ѭ��������Ҫ�ý������õ��ܳ����ֵ��ȷ��
					{
						//�������ǵ�
						short root_x = x_pos;
						short root_y = y_pos;
						//������������߽�����
						x_pos_max = x_pos_max > x_pos ? x_pos_max : x_pos;
						y_pos_max = y_pos_max > y_pos ? y_pos_max : y_pos;
						x_pos_min = x_pos_min < x_pos ? x_pos_min : x_pos;
						y_pos_min = y_pos_min < y_pos ? y_pos_min : y_pos;
						// ѭ���˴� :���ڻ�ȡ��һ��������
						for (counts = 0; counts < 8; counts++)
						{
							// ��ֹ��������  
							curr_d -= curr_d >= 8 ? 8 : 0;
							curr_d += curr_d < 0 ? 8 : 0;
							//��ʵ�ϣ�ֻ��Ҫ�ж�7�������ڵ���Ϣ(���˵�һ��֮��)����count=6ʱ�պ�ѭ������һ��������
							if (first_time && (counts == 6))//qwt 9.04 ��������˵�����һ�β��ܽ���if
							{
								continue;
							}
							//���±�ǵ�root;
							root_x = x_pos + direction_x[curr_d];//����������
							root_y = y_pos + direction_y[curr_d];//����������
							//�жϵ��Ƿ�Խ�磬����ͼ�����������
							if (root_x < 0 || root_x >= devpar.ImgHeight*devpar.PictureNum || root_y < 0 || root_y >= devpar.ImgWidth)
							{
								curr_d++;
								continue;
							}
							//������ڱ�Ե  
							if (255 == src[root_y + root_x * devpar.ImgMakeborderWidth])
							{
								curr_d -= 2;   //���µ�ǰ����  
								Point_counts++;
								//����b_pt:���ٵ�root��  
								x_pos = root_x;
								y_pos = root_y;
								break;   // ����forѭ��  
							}
							curr_d++;
						}   // end for  ��
							//���ٽ��������쳣����
						if (8 == counts || (x_pos >= (x + 8) && y_pos >= (y + 8)))
						{
							break;
						}
						//��������
						if (y_pos == j && x_pos == i)
						{
							//����������Ϣ
							c_length[Id] = Point_counts;
							x_min[Id] = x_pos_min;
							x_max[Id] = x_pos_max;
							y_min[Id] = y_pos_min;
							y_max[Id] = y_pos_max;
							break;
						}//��������if
						//�ж�
						if (cLengthCount == 0)
						{
							first_time = true;
						}
					}//��Χfor����			
				}//�ж�ǰ����if����
				j = y_pos_max > j ? y_pos_max : j;//���º�����������
			}//��һ��for����
			i = x_pos_max > i ? x_pos_max : i;
		}//�ڶ���for ����
	}
}//�˺�������

 /*���������ȡ*/
 //��������getCounter��ȡ�� x_min��x_max�����У� y_min��y_max�����С�
__global__  void GetInfo(unsigned char* src_gray, short *length, short* x_min, short * y_min, short* x_max, short *y_max, short *xpos, short*ypos, short *area, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short count = 0;//�����������
	int sum_gray = 0;//Բ������ĻҶ�ֵ֮��
	int x_sum = 0;//x�Ҷ�ֵ��Ȩ��
	int y_sum = 0;//y�Ҷ�ֵ��Ȩ��
	int mThreshold = devpar.Threshold;//��ֵ����ֵ
									  //���淽λ�б߽�
	short xmm = x_min[Id];
	short xmx = x_max[Id];
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = (ymx - ymm + 3) / 4 * 4;
	unsigned char temp0, temp1, temp2, temp3;//�üĴ����ݴ�ͼ�����ݣ���Сȫ���ڴ�ķ��ʣ���߷ô�Ч��
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	if (length[Id] > devpar.LengthMin)
	{
		//ѭ���Ż�,�������������һЩ�����ֵ����Ҫ����һ�£�
		for (int i = xmm; i <= xmx; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
				//��ֹԽ��
				temp0 = j > ymx ? 0 : 1;  //qwt
				temp1 = j + 1 > ymx ? 0 : 1;
				temp2 = j + 2 > ymx ? 0 : 1;
				temp3 = j + 3 > ymx ? 0 : 1;

				//���ݶ�ֵ����ֵ 
				temp0 *= src_gray[j   *temp0 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
				temp1 *= src_gray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
				temp2 *= src_gray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
				temp3 *= src_gray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] > mThreshold ? src_gray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;

				count += temp0 > 0 ? 1 : 0; //�������
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

//ɸѡ���ظ���Ϣ�ĺ���,����˺���Ҫ����ʧ��
__global__  void GetTrueInfo(short *xcenter, short *ycenter, short*index, short *sArea, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short temp = 0;
	index[Id] = 0;//����������
	if ((Id > devpar.ColThreadNum) && (Id < devpar.ColThreadNum*(devpar.RowThreadNum - 1)))
	{
		if (xcenter[Id] != 0)
		{
			//�ж�һ���̻߳�ȡ�������Ƿ���������ڵ��ҷ��̣߳���+1�����·��̣߳���+1����ȡ������һ�¡�����һ�����������ֵ
			//��
			temp += ((xcenter[Id] == xcenter[Id + 1]) && (ycenter[Id] == ycenter[Id + 1])) ? 1 : 0;//��
			temp += ((xcenter[Id] == xcenter[Id + devpar.ColThreadNum]) && (ycenter[Id] == ycenter[Id + devpar.ColThreadNum])) ? 1 : 0;//��
			temp += ((xcenter[Id] == xcenter[Id - devpar.ColThreadNum + 1]) && (ycenter[Id] == ycenter[Id - devpar.ColThreadNum + 1])) ? 1 : 0;//����
			temp += ((sArea[Id] > devpar.AreaMin) && (sArea[Id] < devpar.AreaMax)) ? 0 : 1;//qwt8-8
			index[Id] = temp > 0 ? 0 : Id;
		}
	}
}

/*����ģʽ��������ȡ*/
//���룺 ��λ��    �Ҷ�ͼ    ����ͼ
//����� �ܳ�  ���  ��������
__global__	void GetRecInfo(RecData* mRec, unsigned char *psrcgray, unsigned char *psrccounter,
	short *length, short* area, short *xpos, short *ypos, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x;
	int mThreshold = devpar.Threshold;//��ֵ����ֵ
	short count = 0;//�����������
	int clengthCount = 0;
	short clength = 0;//�ܳ�����
	int sum_gray = 0;//Բ������ĻҶ�ֵ֮��
	int x_sum = 0;//x�Ҷ�ֵ��Ȩ��
	int y_sum = 0;//y�Ҷ�ֵ��Ȩ��

				  //��ȡ��λ��
	short xmm = mRec[Id].RecXmin;
	short xmx = mRec[Id].RecXmax;
	short ymm = mRec[Id].RecYmin;
	short ymx = mRec[Id].RecYmax;
	short jcount = (ymx - ymm + 3) / 4 * 4;//qwt
	unsigned char temp0, temp1, temp2, temp3;//temp���ڼ������ġ����
	unsigned char t0, t1, t2, t3;//t���ڼ����ܳ�
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	length[Id] = 0;
	//ѭ���Ż�,�������������һЩ�����ֵ����Ҫ����һ�£�
	for (int i = xmm; i <= xmx; i++)
		for (int j = ymm; j <= ymm + jcount; j = j + 4)
		{
			//��ֹԽ��
			temp0 = j    > ymx ? 0 : 1;
			temp1 = j + 1> ymx ? 0 : 1;
			temp2 = j + 2> ymx ? 0 : 1;
			temp3 = j + 3> ymx ? 0 : 1;

			t0 = temp0;//qwt
			t1 = temp1;
			t2 = temp2;
			t3 = temp3;

			//���ݶ�ֵ����ֵ
			temp0 *= psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
			temp1 *= psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
			temp2 *= psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
			temp3 *= psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;

			t0 *= psrccounter[j   *t0 + i * devpar.ImgMakeborderWidth];
			t1 *= psrccounter[(j + 1)*t1 + i * devpar.ImgMakeborderWidth];
			t2 *= psrccounter[(j + 2)*t2 + i * devpar.ImgMakeborderWidth];
			t3 *= psrccounter[(j + 3)*t3 + i * devpar.ImgMakeborderWidth];


			count += temp0 > 0 ? 1 : 0; //�������
			count += temp1 > 0 ? 1 : 0;
			count += temp2 > 0 ? 1 : 0;
			count += temp3 > 0 ? 1 : 0;


			clengthCount += t0 + t1 + t2 + t3;//�ܳ�����
			sum_gray += temp0 + temp1 + temp2 + temp3;


			x_sum += i * temp0 + i * temp1 + i * temp2 + i * temp3;
			y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;
		}
	//ɸѡ����
	clength = clengthCount / 255;
	length[Id] = clength;
	area[Id] = count;
	xpos[Id] = x_sum / sum_gray;
	ypos[Id] = y_sum / sum_gray;
}

//-------------------------------------------------------����----------------------------------------//
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
// 8λ�Ҷ�BMP��ʽͼ���ȡ
bool RmwRead8BitBmpFile2Img(const char * filename, unsigned char*pImg, unsigned char*Binarization, int *width, int *height)
{
	FILE *binFile;
	BITMAPFILEHEADER fileHeader;//�ļ�ͷ
	BITMAPINFOHEADER bmpHeader;//��Ϣͷ
	BOOL isRead = TRUE;
	int ImgDeep;
	int linenum, ex; // nenum:һ�����ص��ֽ���������������ֽ�

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
	ImgDeep = bmpHeader.biBitCount / 8;//ÿ��������ռ�ֽ���Ŀ
	linenum = (*width * ImgDeep + 3) / 4 * 4;//����Ҫ��
	ex = linenum - *width * ImgDeep;   //ÿһ�е�����ֽ�

	fseek(binFile, fileHeader.bfOffBits, SEEK_SET);
	//��ȡ�Ҷ�ͼ
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
	//��ȡλͼ
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
			//bmpת�Ҷ�
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
				//bmpת�Ҷ�
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

//����������ȡ��λ��---��ȡ��λ�����ظ�
void GetImgBoxHost(const char *path)
{
	Parameter devpar;
	//��ʼ��ͼ����Ϣ����
	devpar.ImgHeight = gStructVarible.ImgHeight;
	devpar.ImgWidth = gStructVarible.ImgWidth;
	devpar.Threshold = gStructVarible.Threshold;
	devpar.LengthMin = gStructVarible.LengthMin;
	devpar.LengthMax = gStructVarible.LengthMax;
	devpar.AreaMin = gStructVarible.AreaMin;
	devpar.AreaMax = gStructVarible.AreaMax;
	devpar.PictureNum = gStructVarible.PictureNum;

	//��λ��������
	const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
	//��ʼ��CPU�˷�λ������
	if (gHostRecData.size() != 0)
		gHostRecData.clear();
	//��ʼ��  ��λ�и�������
	gImgXcenter = 0;
	gImgYcenter = 0;
	gXcenterOffset = 0;
	gYcenterOffset = 0;
	//ͼ��ռ����
	unsigned char *ImgHostdata = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum]; //qwt���������BUG
	unsigned char *m_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//��ֵ��ͼ
	unsigned char *n_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//����ͼ
	unsigned char *c_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//����ͼ	
	
	int Picoffset = devpar.ImgHeight * devpar.ImgWidth;//qwt//��ȡͼƬ
	for (int j = 0; j < devpar.PictureNum; j++)
	{
		RmwRead8BitBmpFile2Img(path,NULL, ImgHostdata + j*Picoffset, &devpar.ImgWidth, &devpar.ImgHeight);//qwt823
	}
	//��ֵ��
	for (int i = 0; i <devpar.ImgHeight*devpar.PictureNum; i++)
	{
		for (int j = 0; j < devpar.ImgWidth; j++)
		{
			m_ptr[j + i * devpar.ImgWidth] = ImgHostdata[j + i * devpar.ImgWidth] > devpar.Threshold ? 255 : 0;
			c_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
			n_ptr[j + i * devpar.ImgWidth] = m_ptr[j + i * devpar.ImgWidth];
		}

	}
	//����
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
	//��ʴ  c_ptr������
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
	//��λ��
	short xmax;
	short xmin;
	short ymax;
	short ymin;
	// ��Ե����  
	int i, j, counts = 0, curr_d = 0;//counts����ѭ������   curr_d�Ƿ������������ID
	short cLength;
	//��ȡ��λ����
	for (i = 1; i <devpar.ImgHeight*devpar.PictureNum - 1; i++)
		for (j = 1; j <devpar.ImgWidth - 1; j++)
		{
			// ��ʼ�㼰��ǰ��  
			cv::Point b_pt = cv::Point(i, j);
			cv::Point c_pt = cv::Point(i, j);
			// �����ǰ��Ϊǰ����  
			if (255 == c_ptr[j + i * devpar.ImgWidth])
			{
				cLength = 1;
				xmin = xmax = i;
				ymin = ymax = j;

				bool first_t = false;
				bool tra_flag = false;//���ñ�־λ
									  // ����  
				c_ptr[j + i * devpar.ImgWidth] = 0;    // �ù��ĵ�ֱ�Ӹ�����Ϊ0  

													   // ���и���  
				while (!tra_flag)
				{
					// ѭ���˴�  
					for (counts = 0; counts < 8; counts++)
					{
						// ��ֹ��������  
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

						// ���ٵĹ��̣�Ӧ���Ǹ������Ĺ��̣���Ҫ��ͣ�ĸ���������root��  
						c_pt = cv::Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);

						// �߽��ж�  
						if ((c_pt.x > 0) && (c_pt.x < devpar.ImgHeight*devpar.PictureNum - 1) &&
							(c_pt.y > 0) && (c_pt.y < devpar.ImgWidth - 1))
						{
							// ������ڱ�Ե  
							if (255 == c_ptr[c_pt.x*devpar.ImgWidth + c_pt.y])
							{
								//���°�Χ��
								xmax = xmax > c_pt.x ? xmax : c_pt.x;
								ymax = ymax > c_pt.y ? ymax : c_pt.y;
								xmin = xmin < c_pt.x ? xmin : c_pt.x;
								ymin = ymin < c_pt.y ? ymin : c_pt.y;
								curr_d -= 2;   //���µ�ǰ����  
								c_ptr[c_pt.x*devpar.ImgWidth + c_pt.y] = 0;
								// ����b_pt:���ٵ�root��  
								b_pt.x = c_pt.x;
								b_pt.y = c_pt.y;
								cLength++;
								break;   // ����forѭ��  
							}
						}
						curr_d++;
					}   // end for  
						// ���ٵ���ֹ���������8���򶼲����ڱ�Ե  
					if (8 == counts)
					{
						// ����  
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
	//��ʼ�� ��λ�и�������
	if (gHostRecData.size() > 0)
	{
		//��ȡͼ���־�������ۼӺ�
		for (int k = 0; k < gHostRecData.size(); k++)
		{
			//�������κлҶ����ļ���
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
		//�����Ȩ�Ҷ�����
		gImgXcenter /= gHostRecData.size();
		gImgYcenter /= gHostRecData.size();
		//����ƫ������ʼ��
		gXcenterOffset = 0;
		gYcenterOffset = 0;
		gRectRealNum = gHostRecData.size();//��ȡ��λ��ʵ������ֵ
		//������λ�����������ú����߳�����
		int rRecNum = (gHostRecData.size() + 127) / 128 * 128;
		gHostRecData.resize(rRecNum, RecData{ 0,0,0,0 });
	}
	//�ͷ��ڴ�
	delete[]ImgHostdata;
	delete[]m_ptr;
	delete[]n_ptr;
	delete[]c_ptr;
}

//qwt7.26
class R : public Runnable
{
public:
	Parameter Devpar;//��������
	~R()
	{
	}
	void Run()
	{
		//������
		cudaError_t  err;
		int img_index = 0;
		int Width;
		int Height;
		char strFilename[100];                                          //��1������һ���ַ����鱣��----ͼƬ�Ķ�ȡ·�� 
		char saveFilename[100];                                         //��1������һ���ַ����鱣��----ͼƬ�Ĵ洢·��
		char* path = "E:\\project\\Simulation\\output";


		//��������
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//����Ŀ�ȼ���
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / 8;
		Devpar.ColThreadNum = (Devpar.ImgWidth / 8 + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);

		//����GPU�豸��
		/*������*/
		//����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*�豸��*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//�������ʱ�����ռ䣬�����з�λ�С����������GPU���ڴ��GPU�Դ�
		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//����ܳ�
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//���
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//��������x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//��������y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault));//����������
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//��λ�� xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//�豸�����	�ܳ�
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				���
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short)));//				������
		}

		while ((img_index + CUDAStreams) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rDev_in[i], gHostImage[img_index + i], sizeof(uchar)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ִ�лҶȻ�����ֵ���˺�������
				CopyMakeBorder << <mGrid1, 128, 0, rcS[i] >> > (rDev_in[i], rDev_padding[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ִ�лҶȻ�����ֵ���˺�������
				Binarization << <mGrid1, 128, 0, rcS[i] >> > (rDev_padding[i], rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//�߽���ȡ
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
				//��ȡ�����ͱ�Ե��
				GetCounter << <mGrid2, 128, 0, rcS[i] >> > (rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], Devpar);//��ȡ�����ĺ���
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//��ȡ���������//��ȡ������Ϣ�˺���
				GetInfo << <mGrid2, 128, 0, rcS[i] >> > (rDev_padding[i],  gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
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
				//ɸѡ��ӡ��ȡ������
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
				//�����־������
				if (myInfo.size() > 0) 
				{
					FILE* fp;
					sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
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
			//�豸���ڴ�
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

//����ģʽ����R
class RecR : public Runnable
{
public:
	Parameter Devpar;//��������	
	~RecR()
	{
	}
	void Run()
	{
		cudaError_t err;
		int xCenterSum = 0;//���·�λ���������ݣ�
		int yCenterSum = 0;//���·�λ����������
		int img_index = 0;
		char strFilename[100];                                          //��1������һ���ַ����鱣��----ͼƬ�Ķ�ȡ·�� 
		char saveFilename[100];                                         //��1������һ���ַ����鱣��----ͼƬ�Ĵ洢·��
		char* path = "E:\\project\\Simulation\\output";
		//��������
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//����Ŀ�ȼ���
		int mRecCount = gHostRecData.size();//��ȡ��λ������
		//�˺�������GRid���䣻
		int Gridsize = mRecCount / 128;
		if (Gridsize == 0)//qwt823
			Gridsize = 1;
		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Gridsize, 1, 1);

		/*������*/
		//����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		//���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		//������λ������
		RecData *gpRDevRecData[CUDAStreams];//qwt821
		if (gHostRecData.size() > 0) 
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				checkCudaErrors(cudaMalloc((void**)&gpRDevRecData[i], mRecCount * sizeof(RecData)));//
				cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], mRecCount * sizeof(RecData), cudaMemcpyHostToDevice);
			}
		}
		//�洢�ռ����
		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], mRecCount * sizeof(short), cudaHostAllocDefault));//����ܳ�
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], mRecCount * sizeof(short), cudaHostAllocDefault));//���
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], mRecCount * sizeof(short), cudaHostAllocDefault));//��������x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], mRecCount * sizeof(short), cudaHostAllocDefault));//��������y
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], mRecCount * sizeof(short)));//�豸�����	�ܳ�
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], mRecCount * sizeof(short)));//				���
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
				//ִ�лҶȻ�����ֵ���˺�������
				CopyMakeBorder << <mGrid1, 128, 0, rcS[i] >> > (rDev_in[i], rDev_padding[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ִ�лҶȻ�����ֵ���˺�������
				Binarization << <mGrid1, 128, 0, rcS[i] >> > (rDev_padding[i],  rgpu_2val[i], rgpu_counter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//�߽���ȡ
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
				//��ͬ���еĺ˺�����ͬһGPU����ʱ���Ƿ��Ӱ��˺���������qwt
				GetRecInfo << <mGrid2, 128, 0, rcS[i] >> >(gpRDevRecData[i], rDev_padding[i], rgpu_counter[i],
					gpDevLength[i], gpDevArea[i], gpDevXpos[i], gpDevYpos[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(gpHostLength[i], gpDevLength[i], sizeof(short)*   mRecCount, cudaMemcpyDeviceToHost, rcS[i]);
			}
			//printf(" %s\n", cudaGetErrorString(err));//������
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
		//	printf(" %s\n", cudaGetErrorString(err));//������
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaStreamSynchronize(rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ӡ��ȡ������
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
						xCenterSum += gpHostXpos[i][j];//�����ۼӣ�x
						yCenterSum += gpHostYpos[i][j];//�����ۼӣ�y
						myInfo.push_back(temp);
					}
				}
				//��Χ����Ϣ����
				if (myInfo.size() > 0) 
				{
					gXcenterOffset += (gImgXcenter - xCenterSum / myInfo.size());
					gYcenterOffset += (gImgYcenter - yCenterSum / myInfo.size());
					gImgXcenter = xCenterSum / myInfo.size();
					gImgYcenter = yCenterSum / myInfo.size();
				}
				SignPoint.PointNumbers = myInfo.size();
				//�����־������
				if (myInfo.size() > 0)
				{
					FILE* fp;
					sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
					fp = fopen(strFilename, "wb");
					fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
					fclose(fp);
				}
			}
			img_index += gHostImgblock;
			//���°�Χ��
			if (img_index % 100 == 0)
			{
				for (int i = 0; i < mRecCount; i++)
				{
					if (gHostRecData[i].RecXmin != 0 && gHostRecData[i].RecYmin != 0)
					{
						gHostRecData[i].RecXmin += gXcenterOffset;//CPU�˷�λ��
						gHostRecData[i].RecXmax += gXcenterOffset;
						gHostRecData[i].RecYmin += gYcenterOffset;
						gHostRecData[i].RecYmax += gYcenterOffset;
					}
					//�߽��ж�
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
			//�豸���ڴ�
			cudaFree(gpDevLength[i]);
			cudaFree(gpDevArea[i]);
			cudaFree(gpDevXpos[i]);
			cudaFree(gpDevYpos[i]);
			cudaFree(gpRDevRecData[i]);
		}
	}
};
//����ģʽ����S

//����ԭͼ�������
IMGSIMULATION_API bool SimulationImageTest(const char *path, Infomation *Info) 
{
	cudaError_t  err;
	int mWidth, mHeight;
	gHostPathImgNumber = 20;//����ͼƬ��������
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		err = cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth *gStructVarible.PictureNum* sizeof(unsigned char), cudaHostAllocDefault);
	}
	int Picoffset = gStructVarible.ImgHeight * gStructVarible.ImgWidth;//����ͼƬ��ַƫ����
	for (int i = 0; i < gHostPathImgNumber; i++)
	{
		for (int j = 0; j < gStructVarible.PictureNum; j++)
		{
			RmwRead8BitBmpFile2Img(path, NULL, gHostImage[i]+j*Picoffset, &mWidth, &mHeight);
		}
	}
	//����ͼƬ�Ƿ��ȡ�ɹ�------------------------------------------------------------------------------------------------------------------
	//cv::Mat img(gStructVarible.ImgHeight*gStructVarible.PictureNum, gStructVarible.ImgWidth, CV_8UC1);
	//for (int i = 0; i < gStructVarible.ImgHeight*gStructVarible.PictureNum; i++)
	//{
	//	uchar* data = img.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ��
	//	for (int j = 0; j < gStructVarible.ImgWidth; j++)   //��ѭ��
	//	{
	//		data[j] = gHostImage[10][j + i *   gStructVarible.ImgWidth];
	//	}
	//}
	//imwrite("pic.bmp", img);
	//-------------------------------------------------------------------------------------------------------------------------------------
	if(gStructVarible.RecModelFlag == true)
		 GetImgBoxHost(path);//��ȡ��Χ��
	cout << gHostRecData.size();
	/****  ��������****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	pExecutor->Init(1, ExtractPointThreads, 1);
	R r;
	RecR recr;

	if (gStructVarible.RecModelFlag == false)//ȫͼģʽ
	{
		//�ṹ�帳ֵ
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
	else //����ģʽ
	{
		//�ṹ�帳ֵ
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
	if (gStructVarible.RecModelFlag == false)//ȫͼģʽ
	{
		return false;
	}
	else if(gStructVarible.RecModelFlag == true)//����ģʽ 
	{
		return true;
	}
}


//ȫ���ڴ�����
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
//ȫ���ڴ��ͷ�
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
	//��������
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
	
	////�����ڴ�ռ�
	Memory_application(gStructVarible);
	Infomation *result = new Infomation;
	SimulationImageTest(gStructVarible.ImgReadPath, result);
	delete result;
	//����
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
	//����ԭ��ͼ
	cv::Mat img = cv::imread(gStructVarible.ImgReadPath, cv::IMREAD_COLOR);
	cv::Vec3b pflag(0, 0, 255);
	for (int i = 0; i < num9/gStructVarible.PictureNum; i++)
	{

		img.at<cv::Vec3b>(RInfo[i].xpos%gStructVarible.ImgHeight, RInfo[i].ypos%gStructVarible.ImgWidth) = pflag;
	}

	//���ư�Χ��
	/*for (int i = 0; i < gRectRealNum; i++)
	{
		cv::Point  Rmin(gHostRecData[i].RecYmin + 2, gHostRecData[i].RecXmin + 2);
		cv::Point  Rmax(gHostRecData[i].RecYmax - 2, gHostRecData[i].RecXmax - 2);
		rectangle(img, Rmin, Rmax, cv::Scalar(0, 0, 255));
	}*/
	return 0;
}

/*ͼ������������*/
//int main() 
//{
//	Mat img = imread("E:\\project\\Simulation\\data_gray_img\\4M.bmp",0);
//	Mat img1(2048, 2000, CV_8UC1);
//	for (int i = 0; i < img1.rows; i++)
//	{
//		uchar* data = img.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ��
//		uchar* data1 = img1.ptr<uchar>(i);
//		for (int j = 0; j < img1.cols; j++)   //��ѭ��
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



/*������ͨģʽ*/
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
//	// �߳����ö���
//	dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight, 1);
//	dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);
//	unsigned char *tHostImage;
//	cudaHostAlloc((void**)&tHostImage, Devpar.ImgHeight *  Devpar.ImgWidth * sizeof(unsigned char), cudaHostAllocDefault);
//	RmwRead8BitBmpFile2Img(path,NULL,tHostImage, &Devpar.ImgWidth, &Devpar.ImgHeight);
//	
//	//���Զ���ͼƬ�Ƿ�ɹ�------------------------------------------------------------------------------------------------------------
//	cv::Mat img1(Devpar.ImgHeight, Devpar.ImgWidth, CV_8UC1);
//	for (int i = 0; i < Devpar.ImgHeight; i++)
//	{
//		uchar* data = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ��
//		for (int j = 0; j < Devpar.ImgWidth; j++)   //��ѭ��
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
//	//�豸���Դ�����
//	short *  tDevRecXLeft;
//	short *  tDevRecYLeft;
//	short *  tDevRecXRight;
//	short *  tDevRecYRight;
//	short  *tDevLength;
//	short  *tDevArea;
//	short  *tDevXpos;
//	short  *tDevYpos;
//	short  *tDevIndex;
//	cudaMalloc((void**)&tDevRecXLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//��λ�� xmin
//	cudaMalloc((void**)&tDevRecYLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
//	cudaMalloc((void**)&tDevRecXRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
//	cudaMalloc((void**)&tDevRecYRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
//	cudaMalloc((void**)&tDevLength, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//�豸�����	�ܳ�
//	cudaMalloc((void**)&tDevArea, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				���
//	cudaMalloc((void**)&tDevXpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				xpos
//	cudaMalloc((void**)&tDevYpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				ypos
//	cudaMalloc((void**)&tDevIndex, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				������
//																							//����ռ�����
//	short *  tHostRecXLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecYLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecXRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostRecYRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostLength = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostArea = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostXpos = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostYpos = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	short *  tHostIndex = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
//	//�˺���ִ��
//	cudaMemcpy(tDevImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth, cudaMemcpyHostToDevice);
//	//ִ�лҶȻ�����ֵ���˺�������
//	CopyMakeBorder << <mGrid1, 128 >> > (tDevImage, tDevpad, Devpar);
//	//ִ�лҶȻ�����ֵ���˺�������
//	Binarization << <mGrid1, 128 >> > (tDevpad, tDev2val, tDevcounter, Devpar);
//	//�߽���ȡ
//	Dilation << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
//	cudaMemcpy(tDev2val, tDevcounter, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth, cudaMemcpyDeviceToDevice);
//	Erosion << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
//	//��ȡ�ܳ��Ͱ�Χ��
//	GetCounter << <mGrid2, 128 >> > (tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, Devpar);//��ȡ�����ĺ���																													//����ͼ��Ԥ�����Ƿ�ɹ�
//	GetInfo << <mGrid2, 128 >> > (tDevpad, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevXpos, tDevYpos, tDevArea, Devpar);
//	GetTrueInfo << <mGrid2, 128 >> > (tDevXpos, tDevYpos, tDevIndex, tDevArea, Devpar);
//	//����������
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
//	//�������ͼ-----------------------------------------------------------------------------------------------------------------------
//	uchar* Src_counter = new uchar[Devpar.ImgHeight*Devpar.ImgMakeborderWidth];
//	cv::Mat img_counter(Devpar.ImgHeight, Devpar.ImgMakeborderWidth, CV_8UC1);
//	err = cudaMemcpy(Src_counter, tDevcounter, sizeof(unsigned char)*Devpar.ImgHeight * Devpar.ImgMakeborderWidth, cudaMemcpyDeviceToHost);
//	printf("%s", cudaGetErrorString(err));
//	for (int i = 0; i < Devpar.ImgHeight; i++)
//	{
//		uchar* data = img_counter.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ��
//		for (int j = 0; j < Devpar.ImgMakeborderWidth; j++)   //��ѭ��
//		{
//			data[j] = Src_counter[j + Devpar.ImgMakeborderWidth* i];
//		}
//	}
//	//���Ҷ�ͼת��Ϊ��ͼ ,��Χ�к�Բ�����궼��������img_out_rec����
//	cv::Mat img_out(Devpar.ImgHeight, Devpar.ImgMakeborderWidth, CV_8UC3);
//	cv::cvtColor(img_counter, img_out, cv::COLOR_GRAY2BGR);
//	//----------------------------------------------------------------------------------------------------------------------------------
//	//ɸѡ���
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
//			//���Ʒ�λ��
//			cv::Point  Rmin(tHostRecYLeft[j] - 1, tHostRecXLeft[j] - 1);
//			cv::Point  Rmax(tHostRecYRight[j] + 1, tHostRecXRight[j] + 1);
//			cv::rectangle(img_out, Rmin, Rmax, cv::Scalar(0, 0, 255));
//			img_out.at<cv::Vec3b>(temp.xpos, temp.ypos) = cv::Vec3b(0, 0, 255);
//		}
//	}
//	//д������
//	if (myInfo.size() > 0)
//	{
//		FILE* fp;
//		fp = fopen(strfilename, "wb");
//		fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
//		fclose(fp);
//	}
//	//��ȡ����������������Բ������
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
//		//����ԭ��ͼ
//		for (int i = 0; i < num9; i++)
//		{
//			img_out.at<cv::Vec3b>(RInfo[i].xpos, RInfo[i].ypos) = cv::Vec3b(0, 255, 0);
//		}
//	}
//	fclose(fr);
//	//�ͷ��ڴ�
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

