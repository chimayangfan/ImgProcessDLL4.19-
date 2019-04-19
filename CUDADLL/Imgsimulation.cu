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
#pragma comment( lib, "GdiPlus.lib" )
using namespace Gdiplus;
using namespace std;
using namespace cv;

//�����豸���ܶ���
#define CPUThreads 2
#define CUDAStreams 5
int gHostImgblock = CPUThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
//����ͼƬ��С����block��thread���� 
#define gThreshold 60   //��ֵ������ֵ
#define counterNum 640  //����Ƕ������ȡ������Ϣʱ�����߳�����
#define gLengthMax 300//�ܳ������ֵ
#define gLengthMin 30 //�ܳ�����Сֵ 
const int gImgHeight = 5120;//����
const int gImgWidth = 5120; //����
const int gThreadNum = gImgHeight * gImgWidth / 64;

#define Pretreatment
#ifdef Pretreatment
#define ReadImageNumber 250
unsigned char* gHostImage[ReadImageNumber];
#endif // Pretreatment

unsigned char* rhost_in[CUDAStreams];//ҳ�����ڴ�
unsigned char* rDev_in[CUDAStreams];//�豸�ڴ�
unsigned char* rgpu_2val[CUDAStreams];//��ֵ��ͼ
unsigned char* rgpu_counter[CUDAStreams];//����ͼ����ִ��findcountores֮�������

unsigned char* shost_in[CUDAStreams];//ҳ�����ڴ�
unsigned char* sDev_in[CUDAStreams];//�豸�ڴ�
unsigned char* sgpu_2val[CUDAStreams];//��ֵ��ͼ
unsigned char* sgpu_counter[CUDAStreams];//����ͼ����ִ��findcountores֮�������

dim3 mGrid(20, 5120);//�����������Ϊȫ�ֱ���
dim3 mGrid2(5, 640);

cudaStream_t *rcS;
cudaStream_t *scS;

/*�ҶȻ��Ͷ�ֵ��*/
//�ú˺����߳�����λ  <<<(5,5120),1024>>>  ���ҵĵ���block�����1024, grid��һ��block����ͼ���һ�С�;   srcΪԭͼ��   dstΪ��ֵ��ͼ��  dst2Ϊ�Ҷ�ͼ
__global__ void Graybmp(unsigned char *src_gray, unsigned char *dst_2val, unsigned char *dst_counter)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//����άgrid��һάblock��
	int temp = int(src_gray[Id]);//�Ĵ����������أ���߷ô�Ч��								
	if (Id < gImgWidth*gImgHeight)
	{
		dst_2val[Id] = unsigned char(255 * int(temp>gThreshold));//��ֵ�������ü�������֧�ṹ
		dst_counter[Id] = unsigned char(255 * int(temp>gThreshold));
	}
}

/*��ȡ��������Ե��⣩*/
//����ʮ����4����ʴ��ԭͼ���ص���һ�������߳�����<<<(5,5120),1024>>>
//����-----�����ֵ��ͼ
__global__  void dilation(unsigned char *src, unsigned char *dst)
{
	const int Id_x = threadIdx.x + blockIdx.x *blockDim.x;//Id_x��������Ϣ  Id_y��������Ϣ
	const int Id_y = blockIdx.y;//Id_y��������Ϣ
	int temp;
	if (Id_x > 1 && Id_x < (gImgWidth - 2) && Id_y>1 && Id_y < gridDim.y - 1)
	{
		if (src[Id_x + Id_y * gImgWidth] == 0)
		{
			temp = int(src[Id_x - 1 + (Id_y - 1)*gImgWidth]) + int(src[Id_x + (Id_y - 1)*gImgWidth]) + int(src[Id_x + 1 + (Id_y - 1)*gImgWidth])
				+ int(src[Id_x - 1 + Id_y * gImgWidth]) + int(src[Id_x + 1 + Id_y * gImgWidth]) +
				int(src[Id_x - 1 + (Id_y + 1)*gImgWidth]) + int(src[Id_x + (Id_y + 1)*gImgWidth]) + int(src[Id_x + 1 + (Id_y + 1)*gImgWidth]);//��4��������
			dst[Id_x + Id_y * gImgWidth] = temp > 0 ? 255 : 0;
		}
	}

}


//��ʴ
__global__  void erosion(unsigned char *src, unsigned char *dst)
{
	const int Id_x = threadIdx.x + blockIdx.x *blockDim.x;//Id_x��������Ϣ  Id_y��������Ϣ
	const int Id_y = blockIdx.y;//Id_y��������Ϣ
	int temp;
	//����4����ֵ�Ϳ��ڲ��㣬��ȡ������Ϣ�����ڵ�dst���Ǵ洢��������Ϣ
	if (Id_x > 0 && Id_x < (gImgWidth - 1) && Id_y>0 && Id_y < gridDim.y)
	{
		if (src[Id_x + Id_y * gImgWidth] != 0)
		{
			temp = int(src[Id_x + (Id_y - 1)*gImgWidth]) + int(src[Id_x - 1 + Id_y * gImgWidth]) +
				int(src[Id_x + 1 + Id_y * gImgWidth]) + int(src[Id_x + (Id_y + 1)*gImgWidth]);//��4����ʴ
			dst[Id_x + Id_y * gImgWidth] = temp >= 1020 ? 0 : 255;
		}
	}
}


/*������ȡ*/
//���ð�����׷�ٷ���ȡ�������߳�����Ϊ<<<640��640>>>��   һ���̴߳���16*16��С������������ȡ׼�������±߽�����ԭ�� 
//srcΪ�������飨��Ե���������c_length��ȡ���ܳ�ֵ��   (x_min,y_min)��(x_max,y_max)���ڱ���������������,����x�йص�Ϊ��������y�йصı���������
__global__  void getCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max)
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
	for (int i = x; i < (x + 8); i++)
	{
		for (int j = y; j < (y + 8); j++)
		{
			if (255 == src[j + i * gImgWidth])
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
				bool first_time = true;//�Ƿ�ʱ��һ�λ�ȡ������
				short counts = 0;//����ѭ������
				short curr_d = 0;//������������������ȡֵ0-7��ʾ�������8�����õķ�λ
								 // ���и���  
				for (short cLengthCount = 0; cLengthCount<gLengthMax; cLengthCount++)//�����ѭ��������Ҫ�ý������õ��ܳ����ֵ��ȷ��
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

						//��ʵ�ϣ�ֻ��Ҫ�ж�7�������ڵ���Ϣ(���˵�һ��֮��)
						if (first_time && (counts == 6))
						{
							first_time = false;
							continue;
						}
						//���±�ǵ�root;

						root_x = x_pos + direction_x[curr_d];//����������
						root_y = y_pos + direction_y[curr_d];//����������

															 //�жϵ��Ƿ�Խ�磬����ͼ�����������
						if (root_x < 0 || root_x >= gImgHeight || root_y < 0 || root_y >= gImgWidth)
						{
							curr_d++;
							continue;
						}
						//������ڱ�Ե  
						if (255 == src[root_y + root_x * gImgWidth])
						{
							curr_d -= 2;   //���µ�ǰ����  
							Point_counts++;
							//����b_pt:���ٵ�root��  
							x_pos = root_x;
							y_pos = root_y;
							break;   // ����forѭ��  
						}
						curr_d++;
					}   // end for  

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
				}//��Χfor����			
			}//�ж�ǰ����if����
			j = y_pos_max>j ? y_pos_max : j;//���º�����������
		}//��һ��for����
		i = x_pos_max>i ? x_pos_max : i;
	}//�ڶ���for ����
}//�˺�������

 //��������getCounter��ȡ�� x_min��x_max�����У� y_min��y_max�����С�
 //����˺�����getCounter���õ��߳�������<<<640,640>>>�����ڷ��ص�ַ�����������
__global__  void getInfo(unsigned char* src_gray, unsigned char* src_counter, short *length, short* x_min, short * y_min, short* x_max, short *y_max, short *xpos, short*ypos, short *area)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short count = 0;//�����������
	int sum_gray = 0;//Բ������ĻҶ�ֵ֮��
	int x_sum = 0;//x�Ҷ�ֵ��Ȩ��
	int y_sum = 0;//y�Ҷ�ֵ��Ȩ��
				  //����ÿ��ѭ���ж϶�Ҫ���ʱ߽磬���Ը��üĴ����洢�߽硣
	short xmm = x_min[Id];
	short xmx = x_max[Id];
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = ((ymx - ymm) / 4 + 1) * 4;
	unsigned char temp0, temp1, temp2, temp3;//�üĴ����ݴ�ͼ�����ݣ���Сȫ���ڴ�ķ��ʣ���߷ô�Ч��
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	if (length[Id] > gLengthMin)
	{
		//ѭ���Ż�,�������������һЩ�����ֵ����Ҫ����һ�£�
		for (int i = xmm; i <= xmx; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
				//��ֹԽ��
				temp0 = j > ymx ? 0 : 1;
				temp1 = j > ymx ? 0 : 1;
				temp2 = j > ymx ? 0 : 1;
				temp3 = j > ymx ? 0 : 1;


				temp0 *= src_gray[j + i * gImgWidth];
				temp1 *= src_gray[j + 1 + i * gImgWidth];
				temp2 *= src_gray[j + 2 + i * gImgWidth];
				temp3 *= src_gray[j + 3 + i * gImgWidth];

				count += temp0>0 ? 1 : 0; //�������
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

//ɸѡ���ظ���Ϣ�ĺ���,����˺���Ҫ����ʧ��
__global__  void getTrueInfo(short *xcenter, short *ycenter, short*index)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short temp = 0;
	index[Id] = 0;//����������
	if ((Id > counterNum) && (Id < counterNum*(counterNum - 1)))
	{
		if (xcenter[Id] != 0)
		{
			//�ж�һ���̻߳�ȡ�������Ƿ���������ڵ��ҷ��̣߳���+1�����·��̣߳���+1����ȡ������һ�¡�����һ�����������ֵ
			//��
			temp += ((xcenter[Id] == xcenter[Id + 1]) && (ycenter[Id] == ycenter[Id + 1])) ? 1 : 0;//��
			temp += ((xcenter[Id] == xcenter[Id + 640]) && (ycenter[Id] == ycenter[Id + 640])) ? 1 : 0;//��
			temp += ((xcenter[Id] == xcenter[Id - 639]) && (ycenter[Id] == ycenter[Id - 639])) ? 1 : 0;//����
			index[Id] = temp > 0 ? 0 : Id;
		}
	}
	//�¼ӵ�����ѡȡģʽ
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

//8λ�Ҷ�BMP��ʽͼ���ȡ
unsigned char *RmwRead8BitBmpFile2Img(const char * filename, int *width, int *height) {
	FILE *binFile;
	unsigned char *pImg = NULL;
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER bmpHeader;
	BOOL isRead = TRUE;
	int linenum, ex; //linenum:һ�����ص��ֽ���������������ֽ� 

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
	ex = linenum - *width * 1;         //ÿһ�е�����ֽ�

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

//BMP��ʽͼ��д��
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
	//��������ṹ����
	FileHeader.bfType = ((WORD)('M' << 8) | 'B');
	FileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * 4L;//2��ͷ�ṹ��ӵ�ɫ��
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
	// д���ɫ��
	for (i = 0, p[3] = 0; i<256; i++)
	{
		p[0] = p[1] = p[2] = i; // blue,green,red; //��255 - i��Ҷȷ�ת
		if (fwrite((void *)p, 1, 4, BinFile) != 4) { Suc = false; break; }
	}

	if (extend)
	{
		ex = new unsigned char[extend]; //��������СΪ 0~3
		memset(ex, 0, extend);
	}

	//write data
	for (pCur = pImg + (height - 1)*width; pCur >= pImg; pCur -= width)
	{
		if (fwrite((void *)pCur, 1, width, BinFile) != (unsigned int)width) Suc = false; // ��ʵ������
		if (extend) // ��������� �������0
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
		//������
		cudaError_t  err;
		int img_index = 0;
		int Width;
		int Height;
		char strFilename[100];                                          //��1������һ���ַ����鱣��----ͼƬ�Ķ�ȡ·�� 
		char saveFilename[100];                                         //��1������һ���ַ����鱣��----ͼƬ�Ĵ洢·��
		char* path = "C:\\pic\\img_data";

		//����GPU�豸��
		cudaSetDevice(0);
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

		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//����ܳ�
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//���
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//��������x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//��������y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//����������
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], gThreadNum * sizeof(short)));//��λ�� xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], gThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], gThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], gThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], gThreadNum * sizeof(short)));//�豸�����	�ܳ�
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], gThreadNum * sizeof(short)));//				���
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], gThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], gThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], gThreadNum * sizeof(short)));//				������
		}

		while ((img_index + CUDAStreams) <= gHostPathImgNumber)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMemcpyAsync(rDev_in[i], gHostImage[img_index + i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyHostToDevice, rcS[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ִ�лҶȻ�����ֵ���˺�������
				Graybmp << <mGrid, 256, 0, rcS[i] >> > (rDev_in[i], rgpu_2val[i], rgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//�߽���ȡ
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
				//��ȡ�����ͱ�Ե��
				getCounter << <mGrid2, 128, 0, rcS[i] >> > (rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i]);//��ȡ�����ĺ���
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//��ȡ���������//��ȡ������Ϣ�˺���
				getInfo << <mGrid2, 128, 0, rcS[i] >> > (rDev_in[i], rgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
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
				sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
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
		char strFilename[100];                                          //��1������һ���ַ����鱣��----ͼƬ�Ķ�ȡ·�� 
		char saveFilename[100];                                         //��1������һ���ַ����鱣��----ͼƬ�Ĵ洢·��
		char* path = "C:\\pic\\img_data";

		//����GPU�豸��
		cudaSetDevice(1);
		/*������*/
		//����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		short *gpHostXpos[CUDAStreams];
		short *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*//������ʾ�ô��룬���Գɹ������Ρ�������
		//��ʱ�����������˰�Χ��
		short *rec_xmin[CUDAStreams];
		short *rec_ymin[CUDAStreams];
		short *rec_xmax[CUDAStreams];
		short *rec_ymax[CUDAStreams];
		/*������ʾ�ô��룬���Գɹ������Ρ�������*/
		/*�豸��*/
		short *gpDevRecXLeft[CUDAStreams];
		short *gpDevRecYLeft[CUDAStreams];
		short *gpDevRecXRight[CUDAStreams];
		short *gpDevRecYRight[CUDAStreams];
		//���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		short  *gpDevXpos[CUDAStreams];
		short  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		for (int i = 0; i < CUDAStreams; i++)
		{
			checkCudaErrors(cudaHostAlloc((void**)&gpHostLength[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//����ܳ�
			checkCudaErrors(cudaHostAlloc((void**)&gpHostArea[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//���
			checkCudaErrors(cudaHostAlloc((void**)&gpHostXpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//��������x
			checkCudaErrors(cudaHostAlloc((void**)&gpHostYpos[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//��������y
			checkCudaErrors(cudaHostAlloc((void**)&gpHostIndex[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//����������
			/*//������ʾ�ô��룬���Գɹ������Ρ�������
			//��ʱ�����������˰�Χ��
			checkCudaErrors(cudaHostAlloc((void**)&rec_xmin[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//xmin
			checkCudaErrors(cudaHostAlloc((void**)&rec_ymin[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//ymin
			checkCudaErrors(cudaHostAlloc((void**)&rec_xmax[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//xmax
			checkCudaErrors(cudaHostAlloc((void**)&rec_ymax[i], gThreadNum * sizeof(short), cudaHostAllocDefault));//ymax
			/*������ʾ�ô��룬���Գɹ������Ρ�������*/
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXLeft[i], gThreadNum * sizeof(short)));//��λ�� xmin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYLeft[i], gThreadNum * sizeof(short)));//	    ymin
			checkCudaErrors(cudaMalloc((void**)&gpDevRecXRight[i], gThreadNum * sizeof(short)));//		xmax
			checkCudaErrors(cudaMalloc((void**)&gpDevRecYRight[i], gThreadNum * sizeof(short)));//		ymax
			checkCudaErrors(cudaMalloc((void**)&gpDevLength[i], gThreadNum * sizeof(short)));//�豸�����	�ܳ�
			checkCudaErrors(cudaMalloc((void**)&gpDevArea[i], gThreadNum * sizeof(short)));//				���
			checkCudaErrors(cudaMalloc((void**)&gpDevXpos[i], gThreadNum * sizeof(short)));//				xpos
			checkCudaErrors(cudaMalloc((void**)&gpDevYpos[i], gThreadNum * sizeof(short)));//				ypos
			checkCudaErrors(cudaMalloc((void**)&gpDevIndex[i], gThreadNum * sizeof(short)));//				������
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
				//ִ�лҶȻ�����ֵ���˺�������
				Graybmp << <mGrid, 256, 0, scS[i] >> > (sDev_in[i], sgpu_2val[i], sgpu_counter[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//�߽���ȡ
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
				//��ȡ�����ͱ�Ե��
				getCounter << <mGrid2, 128, 0, scS[i] >> > (sgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i]);//��ȡ�����ĺ���
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//��ȡ���������//��ȡ������Ϣ�˺���
				getInfo << <mGrid2, 128, 0, scS[i] >> > (sDev_in[i], sgpu_counter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecXRight[i], gpDevRecYLeft[i], gpDevRecYRight[i], gpDevXpos[i], gpDevYpos[i], gpDevArea[i]);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
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
			/*//������ʾ�ô��룬���Գɹ������Ρ�������
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
			//����1��
			uchar *img_counter = new uchar[gImgHeight *gImgWidth];
			cudaMemcpy(img_counter, sgpu_counter[i], sizeof(uchar)* gImgHeight *gImgWidth, cudaMemcpyDeviceToHost);

			//printf("%s", cudaGetErrorString(err));
			Mat img_out_counter(gImgWidth, gImgHeight , CV_8UC1);
			for (int j = 0; j < gImgWidth; j++)
			{
			uchar* data = img_out_counter.ptr<uchar>(j);  //��ȡ��i�е��׵�ַ��
			for (int k = 0; k < gImgHeight ; k++)   //��ѭ��
			{
			data[k] = img_counter[k + j * gImgHeight ];
			}
			}
			//����2-��ǵ��Ƿ���ȡ��ȷ��
			Mat img_out_rect(gImgWidth, gImgHeight , CV_8UC3);
			cvtColor(img_out_counter, img_out_rect, COLOR_GRAY2BGR);

			//ɸѡ��ӡ��ȡ������
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
			//������Χ�о��ο�
			cv::Point temptop(rec_ymin[i][j] - 1, rec_xmin[i][j] - 1);
			cv::Point tempdown(rec_ymax[i][j] + 1, rec_xmax[i][j] + 1);
			rectangle(img_out_rect, temptop, tempdown, Scalar(0, 0, 255), 1, 1, 0);
			img_out_rect.at<Vec3b>(gpHostXpos[i][j], gpHostYpos[i][j])[2] = 255;
			}

			}
			//cout << endl << "��������Ŀ" << myInfo.size() << endl;

			//sprintf_s(strFilename, "C:\\pic\\img_write\\%d.bmp", img_index + i + CUDAStreams + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
			//imwrite(strFilename, img_out_counter);
			delete[]img_counter;
			}
			/*������ʾ�ô��룬���Գɹ������Ρ�������*/
			for (int i = 0; i < CUDAStreams; i++)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", path, img_index + i + CUDAStreams + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
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
			/*//������ʾ�ô��룬���Գɹ������Ρ�������
			cudaFreeHost(rec_xmin[i]);
			cudaFreeHost(rec_ymin[i]);
			cudaFreeHost(rec_xmax[i]);
			cudaFreeHost(rec_ymax[i]);
			/*������ʾ�ô��룬���Գɹ������Ρ�������*/
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
/*����ӿں���*/

//��������
//ͼ��·������ʽ���
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
	//ͼ��Ԥ������Ӳ�����������ڴ�
	#ifdef Pretreatment
		char strFilename[100];
		int mWidth;
		int mHeight;
		for (int i = 0; i < ReadImageNumber; i++)
		{
			sprintf_s(strFilename, "%s\\%d.bmp", path, i + 1); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ� 
			checkCudaErrors(cudaHostAlloc((void**)&gHostImage[i], gImgHeight * gImgWidth * sizeof(unsigned char), cudaHostAllocDefault));
			gHostImage[i] = RmwRead8BitBmpFile2Img(strFilename, &mWidth, &mHeight);
		}
	#endif // Pretreatment
}

//����ԭͼ�������
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
	//�ͷ��ڴ�
	for (int i = 0; i < 100; i++)
	{
		cudaFreeHost(gHostImage[i]);
	}
}

//ȫ���ڴ�����
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

//ȫ���ڴ��ͷ�
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