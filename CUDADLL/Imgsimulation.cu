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
#include <helper_cuda.h>//������
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
//�߳������룻
std::mutex gExtrackPointLock;//R���Rec����̰߳�ȫ��
std::mutex gComressReadDataLock;
std::mutex compress_process_lock;//�߳�����
std::mutex compress_write_lock;//�߳�����
std::mutex compress_writeCPU_lock;//�߳�����

//�����豸���ܶ���
#define ExtractPointThreads 2
#define CompressionThreads 2
#define CUDAStreams 5
#define GRAYCompressStreams 5
//����ʣ��洢�ռ���ֵ��GB��
#define DiskRemainingSpaceThreshold 50
//����ͼƬ��С����block��thread���� 
int gHostImgblock = ExtractPointThreads * CUDAStreams;
int gDeviceCount;
int gHostPathImgNumber;
dim3 blocks;												//ѹ��������Ҫ��cuda �ֿ�����
dim3 threads(8, 8);											//ѹ��������Ҫ��block �߳�������
//���洫�νṹ��
Parameter gStructVarible{ NULL,NULL,NULL,8,1,5120,5120,5120,60,30,300,8,640,640,0,99999,2000,5,0,0 ,4 };
//��־����Ϣ�ṹ��
Infomation SignPoint;
//Ӳ�����ýṹ��
HardwareInfo HardwareParam;//Ӳ�����ýṹ��

#define Pretreatment
#ifdef Pretreatment
#define ReadImageNumber 250
#endif // Pretreatment
unsigned char* gHostImage[250] = { NULL };
unsigned char* gHostColorImage[250] = { NULL };

//-------------------------��λ��Model����-----------------------------//
typedef struct
{
	short RecXmin;
	short RecYmin;
	short RecXmax;
	short RecYmax;
}RecData;//��λ�����ݽṹ
vector<RecData> gHostRecData;//CPU��λ����������
int gRecNum;//��λ�������������ƴͼ��͹�����ķ�λ��������
int gSingleImgRecNum;//����ͼ��λ������

/*-------------------------���ݻ������ݶ���-----------------------*/
struct CircleInfo//�����洢�ṹ��(24�ֽ�)
{
	short index;
	short length;
	short area;
	double xpos;
	double ypos;
};

//ʵʱˢ��ͼ��
unsigned char * OnlineRefreshIMG;
//ͨ�ű���
int  BufferBlockIndex[6] = { 0 };//������ˢ�µĴ����������˶��ٴ�600��ͼƬ��
int  Bufferlength;//ÿ���������ĳ���(��Ҫ��ʼ��)
vector<int>gWorkingGpuId;//�����������豸���豸��
bool ExtractPointInitialSuccessFlag[3] = { false };//���ڱ�Ǹ���������Ƿ��ʼ�����
bool ExtractPointSuccess = false;//ʵ�������־λ

//���κи��±�־λ
unsigned char * gRecupImgData = NULL;//������κ����ݸ���ʱ����Ӧ�Ļ�������һ��ͼƬ��С��
bool DevUpdateRec[3] = { false };//���ñ�־λΪtrueʱ����ʾ CPU�˾��κ������Ѿ�������ɣ�GPU����Ҫ����CPU�˾��κ�������GPU�������°�Χ����
bool HostUpdateRec = false; //���ñ�־Ϊtrueʱ��ʾ�����˾��κ������ݸ�����
bool RecupdataInitialSuccessFlag = false;

//���������
unsigned char * gCameraDress=NULL;
unsigned char * gCameraBuffer[6] = { NULL };
bool CameraBufferFull[6] = { false };//����ͨ������̣߳�����ڴ�����׼������

//ҳ���ڴ滺����(����������)
unsigned char * gHostBuffer[4] = { NULL };
bool PageLockBufferEmpty[4] = { true };
bool PageLockBufferWorking[4] = { false };
int PageLockBufferStartIndex[4];

//ѹ��������������ѹ����
unsigned char *gHostComressiongBuffer[4] = { NULL };
bool gComressionBufferEmpty[4] = { true };
bool gComressionBufferWorking[4] = { false };
int  gComressionBufferStartIndex[4];


//--------------------------------------------------------��ʼ---------------------------------------------//
/***********************************************************************************************
Function:       RGBtoYUV(�˺�����
Description:    ֻ�ڲ�ͼѹ����ʹ��
��.bmpͼ���R��G��B����ת��Ϊѹ������Ҫ�����Ⱥ�ɫ�����ݣ���Щ���ݶ����Դ���

Calls:          ��
Input:          unsigned char* dataIn���Դ�ԭͼ���ַ��
int imgHeight��ͼ��߶ȣ�
int imgWidth��ͼ���ȣ�
unsigned int nPitch���������������ݿ�ȣ�

Output:        unsigned char* Y, unsigned char* Cb, unsigned char* Cr���õ������Ⱥ�ɫ�����ݣ�
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

/*--------------------------------ѹ��������Ҫ�Ľṹ��needmemory-------------------------------*/
struct needmemory
{
	Npp16s *pDCT[3] = { 0,0,0 };							//GPU��DCT�任�����ݴ������
	Npp32s DCTStep[3];										//��¼�ֽڶ����DCT�任���ݴ�С
	NppiDCTState *pDCTState;
	Npp8u *pDImage[3] = { 0,0,0 };							//GPU��ͼ���YCbCr����
	Npp32s DImageStep[3];									//��¼�ֽڶ����YCbCr���ݴ�С
	Npp8u *pDScan;											//GPU�ڻ���������ɨ������
	Npp32s nScanSize;										//pDScan�ĳ�ʼ����С
	Npp8u *pDJpegEncoderTemp;								//GPU�ڻ�����������м�����
	size_t nTempSize;										// pDJpegEncoderTemp�Ĵ�С
	Npp32s nScanLength;										//������������pDScan��С

	Npp8u *hpCodesDC[3];									//��׼���������DC��ACֵ�ͱ���
	Npp8u *hpCodesAC[3];
	Npp8u *hpTableDC[3];
	Npp8u *hpTableAC[3];
};

/*--------------------------------ѹ��������Ҫ�Ľṹ��needdata-------------------------------*/
struct  needdata
{
	NppiSize oDstImageSize;									//�����jpgͼƬ��С������ֵ��
	NppiSize aDstSize[3];									//ʵ��ѹ��ͼƬ����Χ
	Npp8u *pdQuantizationTables;							//GPU�еı�׼������
	NppiEncodeHuffmanSpec *apDHuffmanDCTable[3];			// GPU�еĻ�����ֱ����
	NppiEncodeHuffmanSpec *apDHuffmanACTable[3];			// GPU�еĻ�����������
};

struct Pk
{
	int Offest;//ÿ���ļ���ƫ����
	int FileLen;//�ļ�����
				//int FileNameLen;//�ļ������� 
				//char* FileName;//��Ҫ������ļ���
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
	void UnPack(const char* name, const char* save_path);											//���
																									//void Form_total_head();
	void Form_total_head(int one_picture_width, int one_picture_height, int picture_number, int picture_index);



	Pk* concordance;
	fstream file;
	int concordancesize;													//�������С
	int FileNum;															//�ļ�����
	const char* Fname;														//�����ɺ���ļ���

	char* head_cache;														//ͷ�ļ��ܴ�С
	int head_bias;															//ͷ�ļ���ƫ��
	int table_scale;

};

//void Package::Form_one_head(int index, char* Filename, int FileLen)
void Package::Form_one_head(int index, int one_picture_index, int FileLen)
{
	if (index == 0)															//�õ�ÿ���ļ���ƫ��λ��
	{
		concordance[index].Offest = 0;
	}
	else
	{
		concordance[index].Offest = concordance[index - 1].Offest + concordance[index - 1].FileLen;
	}
	//table_scale = table_scale + strlen(Filename) + 1 + 3 * sizeof(int);					//���������С
	table_scale = table_scale + 3 * sizeof(int);
	//concordance[index].FileNameLen = strlen(Filename) + 1;								//�ļ�����С

	//concordance[index].FileName = new char[50];
	//strcpy(concordance[index].FileName, Filename);
	//cout << concordance[index].FileName << endl;
	concordance[index].FileNumber = one_picture_index;
	concordance[index].FileLen = FileLen;
}

void Package::Form_total_head(int one_picture_width, int one_picture_height, int picture_number, int picture_index)
{
	concordancesize = table_scale + 6 * sizeof(int);					//�õ��������С
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

void Package::UnPack(const char *name, const char* save_path)										//���
{
	int one_picture_width, one_picture_height, picture_number, picture_index;
	file.open(name, ios::in | ios::binary);
	file.read((char*)&concordancesize, sizeof(int));					//��ȡ�������С
	file.read((char*)&FileNum, sizeof(int));							//��ȡ�ļ�����

	file.read((char*)&one_picture_width, sizeof(int));
	file.read((char*)&one_picture_height, sizeof(int));
	file.read((char*)&picture_number, sizeof(int));
	file.read((char*)& picture_index, sizeof(int));

	file.seekg(8 + 4 * 4, ios::beg);
	concordance = new Pk[FileNum];
	for (int i = 0; i < FileNum; ++i)									//��ȡ��������������
	{
		file.read((char*)&concordance[i].Offest, sizeof(int));			//��ȡƫ����
		file.read((char*)&concordance[i].FileLen, sizeof(int));			//��ȡ�ļ���С
		file.read((char*)&concordance[i].FileNumber, sizeof(int));
		//file.read((char*)&concordance[i].FileNameLen, sizeof(int));		//��ȡ�ļ�����С


		//concordance[i].FileName = new char[concordance[i].FileNameLen];
		//memset(concordance[i].FileName, 0, sizeof(char)*concordance[i].FileNameLen);//����Ϊ��
		//file.read(concordance[i].FileName, concordance[i].FileNameLen);//��ȡ�ļ���
	}
	fstream file1;
	for (int i = 0; i < FileNum; ++i)
	{
		char arr[1024] = { 0 };
		//sprintf(arr, "%s", concordance[i].FileName);				//������ļ���map��
		sprintf_s(arr, "%s\\%d.jpg", save_path, concordance[i].FileNumber);
		file1.open(arr, ios::out | ios::binary);
		file.seekg(concordancesize + concordance[i].Offest, ios::beg);		//���ļ�
		for (int j = 0; j < concordance[i].FileLen; ++j)					//copy�ļ�
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
	//for (int i = 0; i < FileNum; ++i)//�ͷ��ڴ�
	//{
	//delete[]concordance[i].FileName;
	//}
}

unsigned char* gpHudata;									//�Ҷ�ͼƬѹ��ʱʹ�ã�������ʼ���̶���ɫ��ֵ
unsigned char* gpHvdata;

//-------------------------------------��־����ȡ�˺���----------------------------------------//
/*************************************************
��������: ColorMakeBorder //

��������: �˺�����ͼ��Width�����϶�������Ŀ������䣬��Width�������Ϊ128����������
.         ��������㹫ʽΪ��int imgWidth = (width + 127) / 128 * 128�� //

���������const unsigned char *colorimg ��colorimg��24λ��ɫͼ�����ݣ�
.         Parameter devpar��devpar�ǰ�����ͼ����Ϣ�Ĳ�����  //

���������unsigned char *dst��dst����Width������������ص���ͼ�����ݣ����ĵ������ֵȡֵ0�� //

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)�� Grid��ImgMakeborderWidth/128, ImgHeight,1����
.         GPU��һ���̶߳�Ӧ����һ�����ص�//
*************************************************/
__global__ void   ColorMakeBorder(const unsigned char * colorimg, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//ͼ��������
	const int Id_x = blockIdx.y;//ͼ��������
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
��������: GrayMakeBorder //

��������: �˺�����ͼ��Width�����϶�������Ŀ������䣬��Width�������Ϊ128����������
.         ��������㹫ʽΪ��int imgWidth = (width + 127) / 128 * 128�� //

���������const unsigned char *src ��Src�ǻҶ�ͼ�����ݣ�
.         Parameter devpar��devpar�ǰ�����ͼ����Ϣ�Ĳ�����  //

���������unsigned char *dst��dst����Width������������ص���ͼ�����ݣ����ĵ������ֵȡֵ0�� //

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)�� Grid��ImgMakeborderWidth/128, ImgHeight,1����
.         GPU��һ���̶߳�Ӧ����һ�����ص�//
*************************************************/
__global__ void  GrayMakeBorder(const unsigned char *src, unsigned char *dst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x*blockDim.x;//ͼ��������
	const int Id_x = blockIdx.y;//ͼ��������
	if (Id_y <  devpar.ImgWidth)
	{
		dst[Id_y + Id_x * devpar.ImgMakeborderWidth] = src[Id_y + Id_x * devpar.ImgWidth];
	}
}

/*************************************************
��������: Binarization //

��������: ���������趨��ͼ����ֵ����ͼ����ж�ֵ������ֵ����ֵ������������� Parameter devpar�У�
.		  ������ֵ������ֵʱ�����õ�����ֵ��Ϊ255��������ֵС����ֵʱ�����õ�����ֵ��Ϊ0�� //

���������unsigned char *psrcgray �ǻҶ�ͼ�����ݣ�ʵ��������Ⱥ�ĻҶ�ͼ��
.         Parameter devpar �ǰ�����ͼ����Ϣ������     //

���������unsigned char *pdst2val �Ƕ�ֵ����������ݣ�ʵ�ζ�Ӧ��ֵͼ��
.         unsigned char *pdstcounter �Ƕ�ֵ����������ݸ����� ʵ�ζ�Ӧ����ͼ         //

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)�� Grid��ImgMakeborderWidth/128, ImgHeight,1����
.         GPU��һ���̶߳�Ӧ����һ�����ص� ��    //

*************************************************/
__global__ void Binarization(unsigned char *psrcgray, unsigned char *pdst2val, unsigned char *pdstcounter, Parameter devpar)
{
	const int Id = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;//�̺߳�����
	int temp = int(psrcgray[Id]);//�Ĵ����������أ���߷ô�Ч��								
	if (Id < devpar.ImgMakeborderWidth * devpar.ImgHeight*devpar.PictureNum)//�߽籣��
	{
		pdst2val[Id] = unsigned char(255 * int(temp>devpar.Threshold));//��ֵ��
		pdstcounter[Id] = unsigned char(255 * int(temp>devpar.Threshold));
	}
}

/*************************************************
��������: Dilation  //

��������: �����Զ�ֵ��ͼ����8�������Ͳ���������ĳһ��������ֵΪ0�ĵ�İ��������з�0���ص㣬�򽫸õ���Ϊ255�� //

���������unsigned char *psrc �Ƕ�ֵ��ͼ���ݣ��ò�����������Ϊ��ʴ������ģ�帱����
.         Parameter devpar �ǰ�����ͼ����Ϣ������     //

���������unsigned char *pdst �Ǹ�ʴ������������ݣ�ʵ�ʵ���ʱ���ò��������ֵ��ͼ���ݣ�ͨ�����Ͳ���������и��£�   //

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)�� Grid��ImgMakeborderWidth/128, ImgHeight,1����
.         GPU��һ���̶߳�Ӧ����һ�����ص� ��    //

*************************************************/
__global__  void Dilation(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y����������
	const int Id_x = blockIdx.y;//Id_x��������Ϣ  
	int temp;//��ʱ�����������ۼӰ���������ֵ
	if (Id_y> 1 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x < devpar.PictureNum*devpar.ImgHeight - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] == 0)
		{
			temp = int(psrc[Id_y - 1 + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x - 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x - 1)* devpar.ImgMakeborderWidth])
				+ int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y - 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)* devpar.ImgMakeborderWidth]) + int(psrc[Id_y + 1 + (Id_x + 1)* devpar.ImgMakeborderWidth]);
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp > 0 ? 255 : 0;//���Ͳ���
		}
	}
}

/*************************************************
��������: Erosion  //

��������: ���������Ͳ������ͼ����4����ʴ����������ĳһ��������ֵΪ255�ĵ��4����ʮ�ּ���������0���ص㣬�򽫸õ���Ϊ0�� //

���������unsigned char *psrc �����Ͳ������ͼ�����ݣ�
.         Parameter devpar �ǰ�����ͼ����Ϣ������     //

���������unsigned char *pdst �Ǹ�ʴ������������ݣ�����־������ͼ��
.         ʵ�ʵ���ʱ���ò����������Ͳ������ͼ�����ݣ�ͨ����ʴ����������и��£�//

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)�� Grid��ImgMakeborderWidth/128, ImgHeight,1����
.         GPU��һ���̶߳�Ӧ����һ�����ص� ��    //

*************************************************/
__global__  void Erosion(unsigned char *psrc, unsigned char *pdst, Parameter devpar)
{
	const int Id_y = threadIdx.x + blockIdx.x *blockDim.x;//Id_y��������Ϣ
	const int Id_x = blockIdx.y;//Id_x��������Ϣ
	int temp;//��ʱ�����ۼ�4��������ֵ
			 //����4����ֵ�Ϳ��ڲ��㣬��ȡ������Ϣ�����ڵ�dst���Ǵ洢��������Ϣ
	if (Id_y > 0 && Id_y < (devpar.ImgMakeborderWidth - 1) && Id_x>0 && Id_x <devpar.ImgHeight*devpar.PictureNum - 1)
	{
		if (psrc[Id_y + Id_x * devpar.ImgMakeborderWidth] != 0)
		{
			temp = int(psrc[Id_y + (Id_x - 1)*devpar.ImgMakeborderWidth]) + int(psrc[Id_y - 1 + Id_x * devpar.ImgMakeborderWidth]) +
				int(psrc[Id_y + 1 + Id_x * devpar.ImgMakeborderWidth]) + int(psrc[Id_y + (Id_x + 1)*devpar.ImgMakeborderWidth]);//��4����ʴ
			pdst[Id_y + Id_x * devpar.ImgMakeborderWidth] = temp >= 1020 ? 0 : 255;//��ʴ����
		}
	}
}

/*************************************************
��������: GetCounter  //

��������: ������������ͼ������8����׷�ٷ���ȡ��־����ܳ��Ͱ�Χ�У�  //
.
���������unsigned char *psrc ������ͼ���ݣ�
.         Parameter devpar �ǰ�����ͼ����Ϣ������     //

���������short *c_length ����ȡ�ı�־���ܳ�������ȡʧ��ʱ�����ܳ�������Ϊ0��
.         x_min��y_min��x_max��y_max�Ǳ�־��İ�Χ�����ݣ���Χ����һ�����־�����еľ��Σ���Χ��
.         ���ݰ������ε����Ͻ����꣨x_min��y_min�������½����꣨x_max��y_max����
.		  ��������ȡʧ��ʱ������Χ��������0��

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.		  GPU��һ���̶߳�Ӧ����һ��ͼ��飬һ���߳�������ȡ��һ����־���������Ϣ��
.         �ú˺���������ʱ���߳�����Ϊ�� block(128,1,1)��Grid(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1)��
.		  ����ColThreadNum��RowThreadNum�ֱ��Ƕ�ͼ����зֿ����п������п������Ҷ��з���ͼ�����ĿColThreadNum
.		  ��������䣬���Ϊ��128����������
.         ͼ����Сһ��ΪPicBlockSize��PicBlockSize������PicBlockSizeȡֵһ��Ϊ8��16��32��

*************************************************/
__global__  void GetCounter(unsigned char *src, short *c_length, short* x_min, short * y_min, short* x_max, short *y_max, Parameter devpar)
{
	/*�����������飬���ڸ���������,��ʼ������Ϊ���ҷ���0��λ����˳ʱ����ת45�㣨������1��*/
	const  int direction_y[8] = { 1,1,0,-1,-1,-1,0,1 };
	const  int direction_x[8] = { 0,1,1,1,0,-1,-1,-1 };

	//short Picblocksize = devpar.PicBlockSize;//��ȡͼ����С
	/*��ȡ����������*/
	const int y = (blockIdx.x*blockDim.x + threadIdx.x) * devpar.PicBlockSize;//y����������
	const int x = blockIdx.y * devpar.PicBlockSize;//x����������
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//�̺߳�
																						 /*��ʼ��������ֵ*/
	c_length[Id] = 0;
	x_min[Id] = 0;
	x_max[Id] = 0;
	y_min[Id] = 0;
	y_max[Id] = 0;
	bool SuccessFlag = false;//���ڱ������Ƿ�ɹ�����Ϊtrueʱ����ʾ��ǰ�߳̿��Ѿ��ɹ���ȡ��һ����־������
							 /*��ʼ����Χ������*/
	short Rec_xmx = 0, Rec_xmm = 0;
	short Rec_ymx = 0, Rec_ymm = 0;

	if ((y / devpar.PicBlockSize) < (devpar.ImgWidth / devpar.PicBlockSize) && (x / devpar.PicBlockSize) < (devpar.ImgHeight*devpar.PictureNum / devpar.PicBlockSize))//�߽��ж�
	{
		for (int i = x; i < (x + devpar.PicBlockSize); i++)
		{
			for (int j = y; j < (y + devpar.PicBlockSize); j++)
			{
				if (255 == src[j + i * devpar.ImgMakeborderWidth])
				{
					/*��ʼ����Χ������*/
					Rec_ymx = j;
					Rec_ymm = j;
					Rec_xmx = i;
					Rec_xmm = i;

					/*������ڵ�*/
					short root_x = i;//������
					short root_y = j;//������
					short counts;//����8����ѭ������
					short curr_d = 0;//������������������ȡֵ0-7��ʾ�������8�����õķ�λ

									 /*���и���*/
					for (short cLengthCount = 2; cLengthCount < devpar.LengthMax; cLengthCount++)//
					{
						/*�������ǵ�*/
						short boot_x = root_x;
						short boot_y = root_y;

						/*���·�λ������*/
						Rec_xmx = Rec_xmx > root_x ? Rec_xmx : root_x;
						Rec_ymx = Rec_ymx > root_y ? Rec_ymx : root_y;
						Rec_xmm = Rec_xmm < root_x ? Rec_xmm : root_x;
						Rec_ymm = Rec_ymm < root_y ? Rec_ymm : root_y;

						/*�������ڵ�İ������*/
						for (counts = 0; counts < 8; counts++)
						{
							/*��ֹ��������*/
							curr_d -= curr_d >= 8 ? 8 : 0;
							curr_d += curr_d < 0 ? 8 : 0;

							/*��ʵ�ϣ�ֻ��Ҫ�ж�7�������ڵ���Ϣ(���˵�һ��֮��)����count=6ʱ�պ�ѭ������һ��������*/
							if (cLengthCount >2 && (counts == 6))
							{
								curr_d++;
								continue;
							}

							/*��ȡ�����boot*/
							boot_x = root_x + direction_x[curr_d];//����������
							boot_y = root_y + direction_y[curr_d];//����������

							/*�жϵ��Ƿ�Խ�磬����ͼ�����������*/
							if (boot_x < 0 || boot_x >= devpar.ImgHeight*devpar.PictureNum || boot_y < 0 || boot_y >= devpar.ImgWidth)
							{
								curr_d++;
								continue;
							}
							/*������ڱ�Ե*/
							if (255 == src[boot_y + boot_x * devpar.ImgMakeborderWidth])
							{
								curr_d -= 2;   //���µ�ǰ����  
								root_x = boot_x;//���¸��ڵ�
								root_y = boot_y;
								break;
							}
							curr_d++;
						}   // end for  

							/*�߽������ж�*/
						if (8 == counts || (root_x >= (x + devpar.PicBlockSize) && root_y >= (y + devpar.PicBlockSize)))
						{
							break;
						}
						/*��������*/
						if (root_y == j && root_x == i)
						{
							x_min[Id] = Rec_xmm;
							x_max[Id] = Rec_xmx;
							y_min[Id] = Rec_ymm;
							y_max[Id] = Rec_ymx;
							c_length[Id] = cLengthCount;
							SuccessFlag = true;
							break;
						}//��������if
					}//��Χfor����			
				}//�ж�ǰ����if����
				if (SuccessFlag)
					break;
				j = Rec_ymx > j ? Rec_ymx : j;//�����з�����������
			}//��һ��for����
			if (SuccessFlag)
				break;
			i = Rec_xmx > i ? Rec_xmx : i;//�����з�����������
		}//�ڶ���for ����
	}
}//�˺�������

 /*ɸѡ��λ��*/
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
		if ((float(RecBoxHeight) / float(RecBoxWidth))<1.5&& float((RecBoxHeight) / float(RecBoxWidth)) >0.7)//��λ�г������ôȷ��
		{
			if (Rxmm > 0 && Rymm > 0 && Recxmx[Id] < devpar.ImgHeight*devpar.PictureNum - 1 && Recymx[Id] < devpar.ImgWidth - 1)
			{
				yMidPos = Rymm + RecBoxWidth / 2;//��������
				xMidPos = Rxmm + RecBoxHeight / 2;//��������
				for (int i = -1; i < 2; i++)//�����κ�������9�����Ƿ��е�
				{
					if (xMidPos + 1 < devpar.ImgHeight*devpar.PictureNum&&yMidPos + 1 < devpar.ImgWidth)
					{
						temp1 += ImgCounter[yMidPos - 1 + (xMidPos + i)*devpar.ImgMakeborderWidth];
						temp1 += ImgCounter[yMidPos + (xMidPos + i)*devpar.ImgMakeborderWidth];
						temp1 += ImgCounter[yMidPos + 1 + (xMidPos + i)*devpar.ImgMakeborderWidth];
					}
				}
				for (int i = 0; Rxmm + i <= Rxmm + RecBoxHeight - i; i++)//�ж�Height����
				{
					temp1 += ImgCounter[yMidPos + (Rxmm + i)*devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
					temp1 += ImgCounter[yMidPos + (Rxmm + RecBoxHeight - i)*devpar.ImgMakeborderWidth] > 0 ? 1 : 0;
				}
				for (int i = 0; Rymm + i <= Rymm + RecBoxWidth - i; i++)//�ж�width����
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
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//��ȡ�߳�������
	short temp = 0;//������ʱ���������ڱ�ʾ��ǰ����ȡ�������Ƿ�Ӧ��ɾ��
	if (index[Id] != 0)
	{
		if ((Id > devpar.ColThreadNum) && (Id < devpar.ColThreadNum*(devpar.RowThreadNum - 1)))//�߽��ж�
		{
			if (Recxmm[Id] != 0)//�жϵ�ǰ����ȡ�����Ƿ���Ч
			{
				/*�ж�һ��ͼ����ȡ�������Ƿ���������ڵ���ͼ��飨��+1������ͼ��飨��+1��������ͼ��飨��-1����+1����ȡ������һ��*/
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id + 1])) && (Recymm[Id] == Recymm[Id + 1])) ? 1 : 0;//��
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id + devpar.ColThreadNum])) && (short(Recymm[Id]) == short(Recymm[Id + devpar.ColThreadNum]))) ? 1 : 0;//��
				temp += ((short(Recxmm[Id]) == short(Recxmm[Id - devpar.ColThreadNum + 1])) && (short(Recymm[Id]) == short(Recymm[Id - devpar.ColThreadNum + 1]))) ? 1 : 0;//����
				index[Id] = temp > 0 ? 0 : 1;//���������Ч��־
			}
		}
	}
}

__global__  void GetNonRepeatBox(short *Recxmm, short *Recymm, short*index, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;//�߳�����
	const int y = blockIdx.x*blockDim.x + threadIdx.x;//�п�������
	const int x = blockIdx.y;//�п�������
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
��������: GetInfo  //

��������: ���ݷ�λ����Ϣ������Ҷ�ͼ����ȡ��־�����ĺ��������   //
.
���������unsigned char* src_gray �ǻҶ�ͼ��
.         short *length �� ��ȡ�����ܳ���������length>LengthMin ʱ����ʾ��ȡ���ķ�λ����Ϣ��Ч��
.         x_min��y_min��x_max��y_max�Ǳ�־��İ�Χ�����ݣ�
.         Parameter devpar �ǰ�����ͼ����Ϣ������     //

���������short *xpos��short*ypos �����ûҶ����ķ���ȡ���ı�־���������ꣻ
.         short *area  ����ȡ�������������
.		   ����λ��������Чʱ����short *xpos��short*ypos��short *area����0��

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.		   GPU��һ���̶߳�Ӧ����һ��ͼ���ķ�λ�����ݣ�
.         �ú˺���������ʱ���߳�������GetCounter����һ�£� block(128,1,1)��Grid(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1)��

*************************************************/
__global__  void GetInfo(unsigned char* src_gray, short *index, short* x_min, short * y_min, short* x_max, short *y_max, double *xpos, double*ypos, short *area, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	short myArea = 0;
	double sum_gray = 0;//Բ������ĻҶ�ֵ֮��
	double x_sum = 0;//x�Ҷ�ֵ��Ȩ��
	double y_sum = 0;//y�Ҷ�ֵ��Ȩ��
	short mThreshold = devpar.Threshold;//��ֵ����ֵ
	xpos[Id] = 0;
	ypos[Id] = 0;
	int xRealIndex = 0;
	//���淽λ�б߽�
	short ymm = y_min[Id];
	short ymx = y_max[Id];
	short jcount = (ymx - ymm + 3) / 4 * 4;
	unsigned char temp0, temp1, temp2, temp3;//�üĴ����ݴ�ͼ�����ݣ���Сȫ���ڴ�ķ��ʣ���߷ô�Ч��

	if (index[Id] >0)
	{
		//ѭ���Ż�,�������������һЩ�����ֵ����Ҫ����һ�£�
		for (int i = x_min[Id]; i <= x_max[Id]; i++)
			for (int j = ymm; j <= ymm + jcount; j = j + 4)
			{
				xRealIndex = i%devpar.ImgHeight;
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
				myArea += temp0 > 0 ? 1 : 0;//����ۼ�
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
��������: GetRecInfo  //

��������: ����ģʽ��������ȡ����������Ԥ��ȡ�İ�Χ�����ݡ��Ҷ�ͼ������ͼ����ȡ��־���������Ϣ   //

���������RecData* mRec  Ԥ��ȡ�ķ�λ������
.         unsigned char *psrcgray  �Ҷ�ͼ����
.		  unsigned char *psrccounter ����ͼ����
.	      Parameter devpar   ͼ����Ϣ�ṹ��                          //

���������short *length    �ܳ�����
.         short* area      �������
.         short *xpos, short *ypos    ��������

����ֵ  : ��    //

����˵��: ����Ϊ�˺������������˵��ã��豸��ִ�У�
.		  GPU��һ���̶߳�Ӧ����һ��ͼ���ķ�λ�����ݣ�
.         �˺������߳�����Ϊblock(128,1,1)	Grid(Gridsize, 1, 1);����Gridsize= mRecCount / 128,mRecCountΪԤ��ȡ�İ�Χ������,
.		  ��Ԥ��ȡ��Χ��ʱ���԰�Χ��������������䣬���Ϊ��128��������
*************************************************/
__global__	void GetRecInfo(RecData* mRec, unsigned char *psrcgray, unsigned char *psrccounter,
	short *length, short* area, double *xpos, double *ypos, Parameter devpar)
{
	const int Id = threadIdx.x + blockIdx.x*blockDim.x;//��ȡ�̺߳�
	int mThreshold = devpar.Threshold;//��ֵ����ֵ
	short myArea = 0;//�����������
	int clengthCount = 0;//�����ܳ�����ʱ����
	short clength = 0;//�ܳ�����
	double sum_gray = 0;//Բ������ĻҶ�ֵ֮��
	double x_sum = 0;//x�Ҷ�ֵ��Ȩ��
	double y_sum = 0;//y�Ҷ�ֵ��Ȩ��
	int xRealIndex = 0;
					 /*��ȡ��λ��*/
	short xmm = mRec[Id].RecXmin;
	short xmx = mRec[Id].RecXmax;
	short ymm = mRec[Id].RecYmin;
	short ymx = mRec[Id].RecYmax;
	short jcount = (ymx - ymm + 3) / 4 * 4;//����ѭ����������
	unsigned char temp0, temp1, temp2, temp3;//temp����Ҷ�ͼ��������ʱ�������üĴ�������ͼ�����ݣ���߷����ٶȣ�
	unsigned char t0, t1, t2, t3;//t���ڱ�������ͼ��������ʱ����

								 /*���������ʼ��*/
	area[Id] = 0;
	xpos[Id] = 0;
	ypos[Id] = 0;
	length[Id] = 0;

	for (int i = xmm; i <= xmx; i++)
		for (int j = ymm; j <= ymm + jcount; j = j + 4)
		{
			xRealIndex = i%devpar.ImgHeight;
			/*��ֹԽ��*/
			temp0 = j    > ymx ? 0 : 1;
			temp1 = j + 1> ymx ? 0 : 1;
			temp2 = j + 2> ymx ? 0 : 1;
			temp3 = j + 3> ymx ? 0 : 1;

			t0 = temp0;//qwt
			t1 = temp1;
			t2 = temp2;
			t3 = temp3;

			/*��ȡ��������4�����ص�����ֵ*/
			temp0 *= psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[j   *temp0 + i * devpar.ImgMakeborderWidth] : 0;
			temp1 *= psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 1)*temp1 + i * devpar.ImgMakeborderWidth] : 0;
			temp2 *= psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 2)*temp2 + i * devpar.ImgMakeborderWidth] : 0;
			temp3 *= psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth]>mThreshold ? psrcgray[(j + 3)*temp3 + i * devpar.ImgMakeborderWidth] : 0;

			t0 *= psrccounter[j   *t0 + i * devpar.ImgMakeborderWidth];
			t1 *= psrccounter[(j + 1)*t1 + i * devpar.ImgMakeborderWidth];
			t2 *= psrccounter[(j + 2)*t2 + i * devpar.ImgMakeborderWidth];
			t3 *= psrccounter[(j + 3)*t3 + i * devpar.ImgMakeborderWidth];


			myArea += temp0 > 0 ? 1 : 0; //�������
			myArea += temp1 > 0 ? 1 : 0;
			myArea += temp2 > 0 ? 1 : 0;
			myArea += temp3 > 0 ? 1 : 0;


			clengthCount += t0 + t1 + t2 + t3;//�ܳ�����

			sum_gray += temp0 + temp1 + temp2 + temp3;//�Ҷ��ۼ�
			x_sum += xRealIndex* temp0 + xRealIndex * temp1 + xRealIndex * temp2 + xRealIndex * temp3;
			y_sum += j * temp0 + (j + 1)*temp1 + (j + 2)*temp2 + (j + 3)*temp3;//y�Ҷȼ�Ȩ�ۼ�
		}
	clength = clengthCount / 255;//�����ܳ�
								 /*�������*/
	length[Id] = clength;
	area[Id] = myArea;
	xpos[Id] = x_sum / sum_gray;
	ypos[Id] = y_sum / sum_gray;

}

//-------------------------------------------------------����----------------------------------------//

//-------------------------------------�Ҷ�ͼ��ѹ���˺���----------------------------------------//
/**
* �����洢���е�ֵ�ֽ⣨���뷶Χ��-4096��4095�������������֣���ϵ��ֵӳ�䵽ֵ�Ĵ����У���ȷ����λ��С��
*/
__device__ unsigned int GPUjpeg_huffman_value[8 * 1024];
/**
* H
* huffman�����- ÿһ�ֱ������257����Ա (256 + 1 extra)
* ���ΰ��������ĸ�huffman�����:
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
/* �ڲ�����ʵ�� */
static int ALIGN(int x, int y) {  //ȡy��������
								  // y must be a power of 2.
	return (x + y - 1) & ~(y - 1);
}

/***********************************************************************************************************
/***�������ƣ�write_bitstream
/***�������ܣ���bitstreamд�뵽ͼ��bit��d_JPEGdata��ȥ
/***��    �룺bit_location  ÿ��mcuͼ��Ԫ����õ���bit����ʼ��λ��
/***��    �룺bit_length    ÿ��mcuͼ��Ԫ����õ���bit��λ����
/***��    �룺bit_code      ÿ��mcuͼ��Ԫÿ���������ֱ���õ���huffman����
/***��    ����d_JPEGdata    ���ڴ洢ͼ�����ݱ���õ�������bitstream
/***��    �أ��޷���
************************************************************************************************************/
__device__ void write_bitstream(unsigned int even_code, unsigned int odd_code, int length, int bit_location, int even_code_size, BYTE *d_JPEGdata) {
	//��һ���̵߳����ݱ���д�����ݱ��뻺��ռ�
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
*��ʼ��huffman��������ݣ��γɳ������ݱ����
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

	// ����������ݵ�bitλ��
	unsigned int value_nbits = 0;
	while (absolute) {
		value_nbits++;
		absolute >>= 1;
	}
	//�����ݽ�����ڱ��� (�������ݵ�ֵ���ڸ�λ������룻�������ݵ�bitλ�����ڵ�λ�Ҷ���)
	GPUjpeg_huffman_value[tid] = value_nbits | (value_code << (32 - value_nbits));
}

__device__ static unsigned int
gpuhuffman_encode_value(const int preceding_zero_count, const int coefficient,
	const int huffman_lut_offset) {
	// ��ȡ�������ݵ�huffman����
	const unsigned int packed_value = GPUjpeg_huffman_value[4096 + coefficient];

	// ��packed_value�ֽ�ɱ���ͱ���bitλ����
	const int value_nbits = packed_value & 0xf;
	const unsigned int value_code = packed_value & ~0xf;

	// find prefix of the codeword and size of the prefix
	const int huffman_lut_idx = huffman_lut_offset + preceding_zero_count * 16 + value_nbits;
	const unsigned int packed_prefix = gpujpeg_huffman_gpu_tab[huffman_lut_idx];
	const unsigned int prefix_nbits = packed_prefix & 31;

	// ���ر������ݵı�������ı��볤��
	return (packed_prefix + value_nbits) | (value_code >> prefix_nbits);
}

__global__ static void
gpujpeg_huffman_gpu_encoder_encode_block(BSI16 *d_ydst, int MCU_total, BYTE *d_JPEGdata,
	int *prefix_num, int offset, const int huffman_lut_offset) {
	//�����Ӧ��ͼ��block id��	
	const int block_idx = (blockIdx.y * gridDim.x << 2) + (blockIdx.x << 2) + threadIdx.y;
	if (block_idx >= MCU_total) return;

	__shared__ int Length_count[(THREAD_WARP + 1) * 4];
	d_ydst += block_idx << 6;
	const int load_idx = threadIdx.x * 2;
	int in_even = d_ydst[load_idx];
	const int in_odd = d_ydst[load_idx + 1];

	//��ֱ���������в�ֱ���
	if (threadIdx.x == 0 && block_idx != 0) in_even = in_even - d_ydst[load_idx - 64];
	if (threadIdx.x == 0 && block_idx == 0) in_even = in_even - 64;

	//���㵱ǰ��������ǰ��0�ĸ���
	const unsigned int nonzero_mask = (1 << threadIdx.x) - 1;
	const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC���ݶ������Ƿ�������
	const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
	const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
	const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);

	//���㵱ǰ�߳�ż�������ݱ���ǰ��0�ĸ���
	int zeros_before_even = 2 * (zero_pair_count + threadIdx.x - 32);
	if ((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
		zeros_before_even += 1;
	}

	// true if any nonzero pixel follows thread's odd pixel
	const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;

	// ��������λ��������ǰ��ı��� ,�����������in_even��0����in_oddǰ���0�ĸ���+1
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

	// һ��block�Ľ�����־
	if (0 == ((threadIdx.x ^ 31) | in_odd)) {
		// �����Ҫ��ӽ�����־����zeros_before_odd��ֵ��Ϊ16
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

	//����ÿ��BLOCK�з����������ݸ���
	unsigned int prefix_bitmap = __ballot(bit_length);
	int prefix_count = __popc(prefix_bitmap & nonzero_mask);
	if (bit_length) {
		bl_ptr[prefix_count] = bit_length;
		__syncthreads();
		//����ǰ׺�������
		for (int j = 0; j < prefix_count; j++) {
			code_nbits = code_nbits + bl_ptr[j];
		}
	}
	if (threadIdx.x == 31) {
		prefix_num[block_idx * 3 + offset] = code_nbits;
	}
	//����д�뻺�����ľ����ֽ�λ�ã�ȷ��д��d_JPEGdata��λ��
	BYTE *Write_JPEGdata = d_JPEGdata + (block_idx << 6);
	const int bit_location = code_nbits - bit_length;
	const int byte_restbits = (8 - (bit_location & MASK));
	const int byte_location = bit_location >> SHIFT;
	int write_bytelocation = byte_location;
	//��һ���̵߳����ݱ���д�����ݱ��뻺��ռ�
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
/***�������ƣ�CUDA_RGB2YUV_kernel
/***�������ܣ���λͼ��BMP����GRBģʽת��ΪYUV����ģʽ
/***��    �룺d_bsrc       ԭʼ��λͼ����
/***��    �룺nPitch       �ֽڶ����RGB���ݴ�С
/***��    �룺Size         �ֽڶ����YCrCb���ݴ�С
/***��    ����Y\Cr\Cb      ת�����3����ɫ����
/***��    �أ��޷���
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
/***�������ƣ�work_efficient_PrefixSum_kernel(int *X, int *BlockSum, int InputSize)
/***�������ܣ�ǰ׺��ͼ��㸨����������Ҫ�����ݿ鱻�ֳ�n��С���Ժ���ÿ��С���ǰ׺��
/***��    �룺X        ��Ҫ����ǰ׺��͵�����
/***��    ����BlockSum  ǰ׺���ÿ��С����ܺ�
/***��    ����X           ǰ׺��͵����ݵ����ս��
/***��    �أ��޷���
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
/***�������ƣ�work_efficient_BlockUp_kernel(int *dc_component)
/***�������ܣ�ǰ׺��ͼ��㸨����������Ҫ�����ݿ鱻�ֳ�n��С���Ժ���ÿ��С���ǰ׺��
/***��    �룺BlockSum    ��Ҫ����ǰ׺��͵�����
/***��    ����BlockSum    ǰ׺��͵����ݵ����ս��
/***��    �أ��޷���
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
/***�������ƣ�CUDA_DCT8_kernel
/***�������ܣ��ԻҶ�ԭʼͼ�����ݽ���DCT�任
/***��    �룺X        ��Ҫ����ǰ׺��͵�����
/***��    �룺MCU_total   ��Ҫ����ǰ׺��͵����ݸ���
/***��    ����X           ǰ׺��͵����ݵ����ս��
/***��    �أ��޷���
************************************************************************************************************/
__global__ void CUDA_DCT8_kernel(BSI16 *d_ydst, BYTE *d_bsrc, RIM Size, int *DEV_ZIGZAG, float *DEV_STD_QUANT_TAB_LUMIN) {
	__shared__ float block[512];
	int OffsThreadInRow = (blockIdx.x << 6) + (threadIdx.z << 5) + (threadIdx.y << 3) + threadIdx.x;
	if (OffsThreadInRow >= Size.width) return;
	OffsThreadInRow = OffsThreadInRow - (blockIdx.x << 6);    //32*16����ƫ��
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
/***�������ƣ�Data_codelength_kernel
/***�������ܣ���ɨ������ݽ��б��룬����64��mcu bit�����Ȳ�����scanɨ��
/***int *dc_component ����ÿ��mcu��ֱ������������kernel���洢64��mcu bit���ܳ���
/***int *d_ydst  ���뾭��zigzagɨ��������
/***int *prefix_num�����������д洢ÿ��mcu��bit�����ȵ�ǰ׺��
************************************************************************************************************/
__global__ void Data_codelength_kernel(BSI16 *d_ydst, int MCU_total, BYTE *d_JPEGdata,
	int *prefix_num, int offset, const int huffman_lut_offset) {
	//�����Ӧ��ͼ��block id��	
	const int block_idx = (blockIdx.y * gridDim.x << 2) + (blockIdx.x << 2) + threadIdx.y;
	if (block_idx >= MCU_total) return;

	__shared__ int Length_count[(THREAD_WARP + 1) * 4];
	d_ydst += block_idx << 6;
	const int load_idx = threadIdx.x * 2;
	int in_even = d_ydst[load_idx];
	const int in_odd = d_ydst[load_idx + 1];

	//��ֱ���������в�ֱ���
	if (threadIdx.x == 0 && block_idx != 0) in_even = in_even - d_ydst[load_idx - 64];
	if (threadIdx.x == 0 && block_idx == 0) in_even = in_even - 85;

	//���㵱ǰ��������ǰ��0�ĸ���
	const unsigned int nonzero_mask = (1 << threadIdx.x) - 1;
	const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC���ݶ������Ƿ�������
	const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
	const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
	const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);

	//���㵱ǰ�߳�ż�������ݱ���ǰ��0�ĸ���
	int zeros_before_even = 2 * (zero_pair_count + threadIdx.x - 32);
	if ((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
		zeros_before_even += 1;
	}

	// true if any nonzero pixel follows thread's odd pixel
	const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;

	// ��������λ��������ǰ��ı��� ,�����������in_even��0����in_oddǰ���0�ĸ���+1
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

	// һ��block�Ľ�����־
	if (0 == ((threadIdx.x ^ 31) | in_odd)) {
		// �����Ҫ��ӽ�����־����zeros_before_odd��ֵ��Ϊ16
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

	//����ÿ��BLOCK�з����������ݸ���
	unsigned int prefix_bitmap = __ballot(bit_length);
	int prefix_count = __popc(prefix_bitmap & nonzero_mask);
	if (bit_length) {
		bl_ptr[prefix_count] = bit_length;
		__syncthreads();
		//����ǰ׺�������
		for (int j = 0; j < prefix_count; j++) {
			code_nbits = code_nbits + bl_ptr[j];
		}
	}
	if (threadIdx.x == 31) {
		prefix_num[block_idx + 1] = code_nbits + 8;
	}
	//����д�뻺�����ľ����ֽ�λ�ã�ȷ��д��d_JPEGdata��λ��
	BYTE *Write_JPEGdata = d_JPEGdata + (block_idx << 6);
	const int bit_location = code_nbits - bit_length;
	const int byte_restbits = (8 - (bit_location & MASK));
	const int byte_location = bit_location >> SHIFT;
	int write_bytelocation = byte_location;
	//��һ���̵߳����ݱ���д�����ݱ��뻺��ռ�
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

	//��ֱ�������ͽ�����������Ԥ����
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
	if (tid >= MCU_total) return;                                        //���tid>MCU��������ִ��
	d_JPEGdata = d_JPEGdata + (tid << 6);                                //����֮ǰ����õ��������׵�ַ
	BYTE *JPEG_Writedatalocation = d_JPEGdata + 63;                      //λ�ƺ��BYTEҪд���λ��
	BYTE byte_tmp;
	int length = prefix_num[tid + 1] - prefix_num[tid] - 8;                //�õ�ÿ��MCU��������bit������ռ���ֽ���
	int right_shift = prefix_num[tid] & MASK;                            //�õ�ǰ��MCU��������bit���ڱ�MCU��������bit�������ֽ���ռ��bit��
	int left_shift = 8 - right_shift;                                    //�õ���MCU��������bit�����ֽ���ռ��bit��
	byte_location = (length - 1) >> SHIFT;                                 //�õ���MCU��������bit��β�ֽ�����λ��
	int bit_rest = 8 - length + ((byte_location << SHIFT));
	length = length + right_shift + 8;                                   //�õ���MCU��������bit�������ֽڳ���
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

//-------------------------------------------------------����----------------------------------------//

/*************************************************
��������: RmwRead8BitBmpFile2Img  //

��������: �������洢λ�õ�.bmp��ʽͼ������ڴ��У� //

���������const char * filename ������ͼ���ļ�·����
.         unsigned char* pImg :���24λλͼ��ָ�룻
.		  unsigned char* Binarization :��ŻҶ�ͼ��ָ�룻
.		  int* width :����ͼ��������
.		  int* width :����ͼ��������//

���������unsigned char* pImg ��������ͼ��Ϊ�Ҷ�ͼ����ָ��ָ��NULL��
.         unsigned char* Binarization ��������ͼ��Ϊ24λ��ͼ����ָ��ָ��NULL����//

����ֵ  : bool -- ����ɹ���־λ//

����˵��: ���������ڵ��Խ׶Σ�ʵ�ʹ��������������Ƭ�Ѿ�������ڴ������У�
.         �ú����ڵ���ǰ����Ҫ��Ϊͼ��ָ�����ͼ���С���ڴ�����
.         �ڴ������С(Byte) =  width * height * ImgDeep��    //

*************************************************/
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
		return true;
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
				return true;
			}
		}
	}
	else return false;
}

/*************************************************
��������: RmwWrite8bitImg2BmpFile  //

��������: �������ڴ�λ�õ�.bmp��ʽͼ��д�뵽Ӳ���У� //

���������unsigned char* pImg :��ŻҶ�ͼ��ָ�룻
.		  int* width :ͼ��������
.		  int* width :ͼ��������
.		  const char * filename �����ͼ���ļ�·����//

���������const char * filename ��.bmp��ʽ�Ҷ�ͼ����//

����ֵ  : Suc(bool��) -- д���ɹ���־λ    //

����˵��: ���������ڵ��Խ׶Σ�ʵ�ʹ��������������Ƭ�Ѿ�������ڴ������У�
.         �ú����ڵ���ǰ����Ҫ��Ϊͼ��ָ�����ͼ���С���ڴ�����
.         �ڴ������С(Byte) =  width * height * ImgDeep��    //

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

/*************************************************
��������: GetImgBoxHost  //

��������: Ԥ��ȡ��Χ�к������ھ���ģʽʱ��ҪԤ�Ⱥ���ȡ��Χ�С���������CPU�汾�İ�����׷�ٷ���ȡ����ͼ��İ�Χ�С�
.		  ��ȡ���İ�Χ�����ݱ�����ȫ�ֱ���vector<RecData>gHostRecData�С�
.         ������ʼ���˰�Χ�и�����ص�ȫ�ֱ���//

���������const char * filename -��Ҫ��ȡ��λͼ(*.bmp)�ľ���·��            //

�����������   //

����ֵ  : ��  //

����˵��: ��������ȡ�İ�Χ�����ݱ�����ȫ�ֱ���vector<RecData>gHostRecData�У����Ҷ����� gHostRecData��Ԫ����Ŀ�����˹���,
.         ������ĩβ���Ԫ��0����������Ŀ���Ϊ��128��������//

*************************************************/
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
	devpar.RecPadding = gStructVarible.RecPadding;
	//��λ��������
	const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
	//��ʼ��CPU�˷�λ������
	if (gHostRecData.size() != 0)
		gHostRecData.clear();
	//ͼ��ռ����
	unsigned char *ImgHostdata = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum]; //qwt���������BUG
	unsigned char *m_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//��ֵ��ͼ
	unsigned char *n_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//����ͼ
	unsigned char *c_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//����ͼ	
	unsigned char *temp_ptr = new unsigned char[devpar.ImgWidth* devpar.ImgHeight*devpar.PictureNum];//��ʱ����ͼ
																									 //��ȡͼƬ
	int Picoffset = devpar.ImgHeight * devpar.ImgWidth;
	for (int j = 0; j < devpar.PictureNum; j++)
	{
		RmwRead8BitBmpFile2Img(path, NULL, ImgHostdata + j*Picoffset, &devpar.ImgWidth, &devpar.ImgHeight);
	}
	//��ֵ��
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
					temp_ptr[j + i * devpar.ImgWidth] = 255;
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
					temp_ptr[j + i * devpar.ImgWidth] = 0;
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

				bool tra_flag = false;//���ñ�־λ
				c_ptr[j + i * devpar.ImgWidth] = 0;// �ù��ĵ�ֱ�Ӹ�����Ϊ0  
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
						// ���ٵĹ��̣��Ǹ������Ĺ��̣���Ҫ��ͣ�ĸ���������root��  
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
						if (cLength < devpar.LengthMax && (cLength > devpar.LengthMin))
						{
							RecData tempRecData;
							int tempcount = 0;
							if (0.7<double(xmax - xmin) / double(ymax - ymin) < 1.5)//��/��
							{

								//����ͼ���ĵ�9�����ж�
								for (int k = -1; k < 2; k++)
								{
									if ((xmax + xmax) / 2 < devpar.ImgHeight*devpar.PictureNum && (ymax + ymin) / 2 < devpar.ImgWidth)
									{
										tempcount += temp_ptr[(ymax + ymin) / 2 - 1 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
										tempcount += temp_ptr[(ymax + ymin) / 2 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
										tempcount += temp_ptr[(ymax + ymin) / 2 + 1 + ((xmax + xmin) / 2 + i)*devpar.ImgMakeborderWidth];
									}
								}
								//����������-���ж�
								for (int k = xmin; k <= xmax; k++)//�ж�Height����
								{
									tempcount += temp_ptr[(ymax + ymin) / 2 + k*devpar.ImgWidth] > 0 ? 1 : 0;
								}
								for (int k = ymin; k <= ymax; k++)//�ж�width����
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
	//������λ�����������ú����߳�����
	gSingleImgRecNum = gHostRecData.size() / devpar.PictureNum;//���ǵ���ͼ��λ�е�ʵ������
	int rRecNum = (gHostRecData.size() + 127) / 128 * 128;
	gHostRecData.resize(rRecNum, RecData{ 0,0,0,0 });
	gRecNum = rRecNum;//��Χ������
					  //�ͷ��ڴ�
	delete[]ImgHostdata;
	delete[]m_ptr;
	delete[]n_ptr;
	delete[]c_ptr;
	delete[]temp_ptr;
}

//-----------------------------------------���ܴ�����---------------------------------------//
//--------------------------------------------��ʼ------------------------------------------//
/*----------------------------------ȫͼģʽ��־����ȡ������------------------------------*/
class SIM : public Runnable
{
public:
	HardwareInfo HardwarePar;//Ӳ������
	Parameter Devpar;//ͼ�����
	~SIM()//��������
	{
	}
	void Run()
	{
		//����GPU�豸��
		cudaSetDevice(HardwarePar.GpuId);
		//������
		cudaError_t  err, err1;
		clock_t start, end;
		clock_t startp, overp;
		clock_t time3;
		/*��ȡ��ǰ�̺߳�*/

		/***********/
		int img_index;
		char DataFilename[100];
		char strFilename[100];
		const char* path = Devpar.DataReadPath;
		int OutPutInitialIndex = 0; //�����Bin�ļ���ʼ������
		int BufferIndex = 0;//ҳ������������
		long long  Bufferoffset = 0;//������ƫ����
		bool DatafullFlag = false;//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�

		/*----------------------��������------------------------------------------*/
		Devpar.ImgChannelNum = Devpar.ImgBitDeep / 8;//λ��ת����ͨ����
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//����Ŀ�ȼ���
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / Devpar.PicBlockSize;
		Devpar.ColThreadNum = (Devpar.ImgWidth / Devpar.PicBlockSize + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);


		/*----------------------�ڴ�����------------------------------------------*/
		//����CUDA��
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/****  ͼ������  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//�豸�ڴ�
		unsigned char* DevPadding[CUDAStreams];//���߽���ͼ���ڴ�   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//��ֵ��ͼ
		unsigned char* DevCounter[CUDAStreams];//����ͼ����ִ��findcountores֮�������
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight * Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char));  //qwt7.26
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/*������*/
		//����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*�豸��*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//�������ʱ�����ռ䣬�����з�λ�С����������GPU���ڴ��GPU�Դ�
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//����ܳ�
			cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//���
			cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//��������x
			cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//��������y
			cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//����������
			cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//��λ�� xmin
			cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//	    ymin
			cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//		xmax
			cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//		ymax
			cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//�豸�����	�ܳ�
			cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//				���
			cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double) * 2);//				xpos
			cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double) * 2);//				ypos
			err = cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short) * 2);//				������
		}

		//��־����ȡ��������
		while ((img_index + CUDAStreams) <= gHostPathImgNumber && gStructVarible.TerminateFlag == 0)
		{
			//��ͼ������Ϊ�Ҷ�ͼ-����ͨ������ֱ�ӽ����ݿ�����DevPicGray
			if (Devpar.ImgChannelNum == 1)
			{
				for (int i = 0; i < CUDAStreams; i++)
				{
					Bufferoffset = long long(img_index + i)* Devpar.ImgHeight * Devpar.ImgWidth;
					cudaMemcpyAsync(DevPicGray[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ִ�лҶȻ�����ֵ���˺�������
					GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
				}
			}
			else if (Devpar.ImgChannelNum == 3)//��ͼ������Ϊ��ɫͼ-����ͨ������ֱ�ӽ����ݿ�����DevPicColor
			{
				for (int i = 0; i < CUDAStreams; i++)
				{
					Bufferoffset = long long(img_index + i)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
					cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
				}
				for (int i = 0; i < CUDAStreams; i++)//ת�Ҷ�+padding
				{
					ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
				}
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ִ�лҶȻ�����ֵ���˺�������
				Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//�߽���ȡ
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
				//��ȡ�����ͱ�Ե��
				GetCounter << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], Devpar);//��ȡ�����ĺ���
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
				SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
				SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//ɸѡ��ȡ������������ķ��ظ���Ϣ
				GetNonRepeatBox << <mGrid2, 128, 0, CStreams[i] >> > (gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevIndex[i], Devpar);
			}
			for (int i = 0; i < CUDAStreams; i++)
			{
				//��ȡ���������//��ȡ������Ϣ�˺���
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
					sprintf_s(DataFilename, "%s\\%d.bin", Devpar.DataReadPath, img_index + HardwarePar.DeviceID * HardwarePar.CUDAStreamNum + i + 1); //��3����ͼƬ��·������̬��д�뵽DataFilename�����ַ���ڴ�ռ�
					fp = fopen(DataFilename, "wb");
					fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
					fclose(fp);
				}
			}
			img_index += HardwarePar.DeviceCount * HardwarePar.CUDAStreamNum;
		}

		/****       �����ô���       ****/
		/****  ���ڲ����ֶ�ֹͣλ��  ****/
		if (gStructVarible.TerminateFlag == 1)
		{
			char buffer[20];
			sprintf_s(buffer, "%s%d", "img_index = ", img_index);
			FILE* fp;
			sprintf_s(DataFilename, "%s\\%d.txt", Devpar.DataReadPath, 0); //��3����ͼƬ��·������̬��д�뵽DataFilename�����ַ���ڴ�ռ�
			fp = fopen(DataFilename, "wb");
			fwrite(buffer, sizeof(char) * 20, 1, fp);
			fclose(fp);
		}
		/**********************/

		//�ͷ��ڴ�
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
			cudaStreamDestroy(CStreams[i]);
		}
	}
};

class R : public Runnable
{
public:
	Parameter Devpar;//��������
	HardwareInfo HardwarePar;//Ӳ������
	static int  mRindex;
	~R()
	{
	}
	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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
		//����GPU�豸��
		cudaSetDevice(HardwarePar.GpuId);
		//������
		/***********/
		int img_index;
		char strFilename[100];
		const char* path = Devpar.DataReadPath;
		int OutPutInitialIndex = 0; //�����Bin�ļ���ʼ������
		int BufferIndex = 0;//ҳ������������
		long long  Bufferoffset = 0;//������ƫ����
		bool DatafullFlag = false;//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�

		/*----------------------��������------------------------------------------*/
		Devpar.ImgChannelNum =  Devpar.ImgBitDeep / 8;//λ��ת����ͨ����
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//����Ŀ�ȼ���
		Devpar.RowThreadNum = Devpar.ImgHeight*Devpar.PictureNum / Devpar.PicBlockSize;//������ܻ���BUG-���߶Ȳ���PicBlock��������ʱ�����ܳ�������
		Devpar.ColThreadNum = (Devpar.ImgWidth / Devpar.PicBlockSize + 127) / 128 * 128;

		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);


		/*----------------------�ڴ�����------------------------------------------*/
		//����CUDA��
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/****  ͼ������  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//�豸�ڴ�
		unsigned char* DevPadding[CUDAStreams];//���߽���ͼ���ڴ�   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//��ֵ��ͼ
		unsigned char* DevCounter[CUDAStreams];//����ͼ����ִ��findcountores֮�������
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight * Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char)); 
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/*������*/
		//����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];
		short *gpHostIndex[CUDAStreams];
		/*�豸��*/
		short *  gpDevRecXLeft[CUDAStreams];
		short *  gpDevRecYLeft[CUDAStreams];
		short *  gpDevRecXRight[CUDAStreams];
		short *  gpDevRecYRight[CUDAStreams];
		//���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		short  *gpDevIndex[CUDAStreams];

		//�������ʱ�����ռ䣬�����з�λ�С����������GPU���ڴ��GPU�Դ�
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//����ܳ�
			cudaHostAlloc((void**)&gpHostArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//���
			cudaHostAlloc((void**)&gpHostXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//��������x
			cudaHostAlloc((void**)&gpHostYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double), cudaHostAllocDefault);//��������y
			cudaHostAlloc((void**)&gpHostIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short), cudaHostAllocDefault);//����������
			cudaMalloc((void**)&gpDevRecXLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//��λ�� xmin
			cudaMalloc((void**)&gpDevRecYLeft[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
			cudaMalloc((void**)&gpDevRecXRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
			cudaMalloc((void**)&gpDevRecYRight[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
			cudaMalloc((void**)&gpDevLength[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//�豸�����	�ܳ�
			cudaMalloc((void**)&gpDevArea[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				���
			cudaMalloc((void**)&gpDevXpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				xpos
			cudaMalloc((void**)&gpDevYpos[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				ypos
			cudaMalloc((void**)&gpDevIndex[i], Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				������
		}

		ExtractPointInitialSuccessFlag[HardwarePar.DeviceID] = true;

		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			vector<CircleInfo>myInfo;
			img_index = 0;//ͼ�����
			Bufferoffset = 0;//ҳ���ڴ�ƫ��
			
            //�󶨻�����
			while (true)
			{
				gExtrackPointLock.lock();
				mRindex = mRindex % (HardwareParam.DeviceCount + 1);
				if (PageLockBufferEmpty[mRindex] == false && PageLockBufferWorking[mRindex] == false)
				{
					PageLockBufferWorking[mRindex] = true;//��ҳ���ڴ��־λ��Ϊ����״̬--���а�
					OutPutInitialIndex = PageLockBufferStartIndex[mRindex] * Bufferlength;//��ȡͼ��������
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
			//��������
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength) //qwt
				{
					gExtrackPointLock.lock();
					PageLockBufferWorking[BufferIndex] = false;//�������--working��־λ��Ϊfalse
					gExtrackPointLock.unlock();
					PageLockBufferEmpty[BufferIndex] = true;  //
					DatafullFlag = false;
					break;
				}
				//��ͼ������Ϊ�Ҷ�ͼ-����ͨ������ֱ�ӽ����ݿ�����DevPicGray
				if (Devpar.ImgChannelNum == 1)
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)* Devpar.ImgHeight * Devpar.ImgWidth;
						cudaMemcpyAsync(DevPicGray[i], gHostBuffer[BufferIndex] + Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)
					{
						//ִ�лҶȻ�����ֵ���˺�������
						GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
					}
				}
				else if (Devpar.ImgChannelNum == 3)//��ͼ������Ϊ��ɫͼ-����ͨ������ֱ�ӽ����ݿ�����DevPicColor
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
						cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + +Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)//ת�Ҷ�+padding
					{
						ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
					}
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ִ�лҶȻ�����ֵ���˺�������
					Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//�߽���ȡ
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
					//��ȡ�����ͱ�Ե��
					GetCounter << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], Devpar);//��ȡ�����ĺ���
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ɸѡ��ȡ������������ķ��ظ���Ϣ
					SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ɸѡ��ȡ������������ķ��ظ���Ϣ
					SelectTrueBox << <mGrid2, 128, 0, CStreams[i] >> > (DevCounter[i], gpDevLength[i], gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevRecXRight[i], gpDevRecYRight[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ɸѡ��ȡ������������ķ��ظ���Ϣ
					GetNonRepeatBox << <mGrid2, 128, 0, CStreams[i] >> > (gpDevRecXLeft[i], gpDevRecYLeft[i], gpDevIndex[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//��ȡ���������//��ȡ������Ϣ�˺���
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
						headInfo.index = OutPutInitialIndex + img_index + i;//��Ӧ�ļ�����
						headInfo.xpos = 99999;
						headInfo.ypos = 99999;//xpos �� ypos��Ϊͷ��־λ
						headInfo.area = 0;	  //areaΪ0Ҳ��Ϊ������־λ
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
						myInfo[headpos].length = hostindex;//������λ
					}
				}
				img_index += HardwarePar.CUDAStreamNum*Devpar.PictureNum;
			}
			//	д����
			if (myInfo.size() > 0)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", path, OutPutInitialIndex); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
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
			cudaStreamDestroy(CStreams[i]);
		}
	}
};
int R::mRindex = 0;//��̬������ʼ��

/*----------------------------------����ģʽ��־����ȡ������------------------------------*/
class RecR : public Runnable
{
public:
	HardwareInfo HardwarePar;//Ӳ������
	Parameter Devpar;//��������	
	static int  mRecindex;
public:
	~RecR()//��������
	{
	}
	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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

		//����GPU�豸��
		cudaSetDevice(HardwarePar.GpuId);
		//��������
		char DataFilename[100]; //����һ���ַ����鱣��----ͼƬ�Ķ�ȡ·�� 
		int img_index = 0;//���ͼ�� bin����
		int OutPutInitialIndex = 0; //�����Bin�ļ���ʼ������
		int BufferIndex = 0;//ҳ������������
		long long  Bufferoffset = 0;//������ƫ����
		bool DatafullFlag = false;//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�
		const char* path = Devpar.DataReadPath;

		/*----------------------��������------------------------------------------*/
		Devpar.ImgChannelNum = Devpar.ImgBitDeep / 8;//λ��ת����ͨ����
		Devpar.ImgMakeborderWidth = (Devpar.ImgWidth + 127) / 128 * 128;//����Ŀ�ȼ���
		int Gridsize = gRecNum / 128;
		if (Gridsize == 0)//qwt823
			Gridsize = 1;
		/****  �˺���Grid  ****/
		dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
		dim3 mGrid2(Gridsize, 1, 1);

		/*----------------------�洢���ռ�����------------------------------------------*/
		//����CUDA��
		cudaStream_t *CStreams;
		CStreams = (cudaStream_t *)malloc(CUDAStreams * sizeof(cudaStream_t));

		/***  ͼ������  ****/
		unsigned char* DevPicColor[CUDAStreams];
		unsigned char* DevPicGray[CUDAStreams];//�豸�ڴ�
		unsigned char* DevPadding[CUDAStreams];//���߽���ͼ���ڴ�   qwt7.26
		unsigned char* Dev2Val[CUDAStreams];//��ֵ��ͼ
		unsigned char* DevCounter[CUDAStreams];//����ͼ����ִ��findcountores֮�������
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaStreamCreate(&(CStreams[i]));
			cudaMalloc((void**)&DevPicColor[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPicGray[i], Devpar.ImgHeight * Devpar.ImgWidth*Devpar.PictureNum * sizeof(unsigned char));
			cudaMalloc((void**)&DevPadding[i], Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum * sizeof(unsigned char));  //qwt7.26
			cudaMalloc((void**)&Dev2Val[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
			cudaMalloc((void**)&DevCounter[i], sizeof(unsigned char) * Devpar.ImgHeight * Devpar.ImgMakeborderWidth * Devpar.PictureNum);
		}
		/****  ������  ****/
		//��־����Ϣ����
		short *gpHostLength[CUDAStreams];
		short *gpHostArea[CUDAStreams];
		double *gpHostXpos[CUDAStreams];
		double *gpHostYpos[CUDAStreams];

		/****  �豸��  ****/
		//��־����Ϣ���
		short  *gpDevLength[CUDAStreams];
		short  *gpDevArea[CUDAStreams];
		double  *gpDevXpos[CUDAStreams];
		double  *gpDevYpos[CUDAStreams];
		RecData *gpRDevRecData[CUDAStreams];//qwt821
	    //������λ������
		if (gRecNum > 0)
		{
			for (int i = 0; i < CUDAStreams; i++)
			{
				cudaMalloc((void**)&gpRDevRecData[i], gRecNum * sizeof(RecData) * 2);//�������2�������ǣ���λ�п�����ʵ���ڼ���ĿҪ�䣬���ܻ���һ�㣬��ֹ����֮���ڴ�Խ��
				cudaMemcpy(gpRDevRecData[i], &gHostRecData[0], gRecNum * sizeof(RecData), cudaMemcpyHostToDevice);
			}
		}

		//�������ʱ�����ռ䣬�����з�λ�С����������GPU���ڴ��GPU�Դ�
		for (int i = 0; i < CUDAStreams; i++)
		{
			cudaHostAlloc((void**)&gpHostLength[i], gRecNum * sizeof(short), cudaHostAllocDefault);//����ܳ�
			cudaHostAlloc((void**)&gpHostArea[i], gRecNum * sizeof(short), cudaHostAllocDefault);//���
			cudaHostAlloc((void**)&gpHostXpos[i], gRecNum * sizeof(double), cudaHostAllocDefault);//��������x
			cudaHostAlloc((void**)&gpHostYpos[i], gRecNum * sizeof(double), cudaHostAllocDefault);//��������y
			cudaMalloc((void**)&gpDevLength[i], gRecNum * sizeof(short));//�豸�����	�ܳ�
			cudaMalloc((void**)&gpDevArea[i], gRecNum * sizeof(short));//				���
			cudaMalloc((void**)&gpDevXpos[i], gRecNum * sizeof(double));//				xpos
			cudaMalloc((void**)&gpDevYpos[i], gRecNum * sizeof(double));//				ypos
		}

		ExtractPointInitialSuccessFlag[HardwarePar.DeviceID] = true;

		//��־����ȡ��������
		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			vector<CircleInfo>myInfo;
			img_index = 0;//ͼ�����
			Bufferoffset = 0;//ҳ���ڴ�ƫ��
							 //������
			while (true)
			{
				gExtrackPointLock.lock();
				mRecindex = mRecindex % (HardwareParam.DeviceCount + 1);
				if (PageLockBufferEmpty[mRecindex] == false && PageLockBufferWorking[mRecindex] == false)
				{
					PageLockBufferWorking[mRecindex] = true;//��ҳ���ڴ��־λ��Ϊ����״̬--���а�
					OutPutInitialIndex = PageLockBufferStartIndex[mRecindex] * Bufferlength;//��ȡͼ��������
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
			//��ȡ����
			while (DatafullFlag)
			{
				if (img_index >= Bufferlength) //qwt
				{
					gExtrackPointLock.lock();
					PageLockBufferWorking[BufferIndex] = false;//�������--working��־λ��Ϊfalse
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
						//ִ�лҶȻ�����ֵ���˺�������
						GrayMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicGray[i], DevPadding[i], Devpar);
					}
				}
				else if (Devpar.ImgChannelNum == 3)//��ͼ������Ϊ��ɫͼ-����ͨ������ֱ�ӽ����ݿ�����DevPicColor
				{
					for (int i = 0; i < CUDAStreams; i++)
					{
						Bufferoffset = long long(img_index + i*Devpar.PictureNum)*Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum;
						cudaMemcpyAsync(DevPicColor[i], gHostBuffer[BufferIndex] + +Bufferoffset, sizeof(unsigned char)* Devpar.ImgHeight * Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice, CStreams[i]);
					}
					for (int i = 0; i < CUDAStreams; i++)//ת�Ҷ�+padding
					{
						ColorMakeBorder << <mGrid1, 128, 0, CStreams[i] >> > (DevPicColor[i], DevPadding[i], Devpar);
					}
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//ִ�лҶȻ�����ֵ���˺�������
					Binarization << <mGrid1, 128, 0, CStreams[i] >> > (DevPadding[i], Dev2Val[i], DevCounter[i], Devpar);
				}
				for (int i = 0; i < CUDAStreams; i++)
				{
					//�߽���ȡ
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
					//��ͬ���еĺ˺�����ͬһGPU����ʱ���Ƿ��Ӱ��˺���������qwt
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
						headInfo.index = OutPutInitialIndex + img_index + i;//��Ӧ�ļ�����
						headInfo.xpos = 99999;
						headInfo.ypos = 99999;//xpos �� ypos��Ϊͷ��־λ
						headInfo.area = 0;	  //areaΪ0Ҳ��Ϊ������־λ
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
						myInfo[headpos].length = hostindex;//������λ
					}
				}
				img_index += HardwarePar.CUDAStreamNum*Devpar.PictureNum;
			}
			//д����
			if (myInfo.size() > 0)
			{
				FILE* fp;
				sprintf_s(DataFilename, "%s\\%d.bin", path, OutPutInitialIndex); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
				fp = fopen(DataFilename, "wb");
				fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
				fclose(fp);
			}
			//���°�Χ��
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
			//�豸���ڴ�
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

/*----------------------------------���κи�����------------------------------------------*/
class RecUpData : public Runnable
{

public:
	Parameter Devpar;//��������	
	~RecUpData()
	{
	}
	void Run()
	{

		char strFilename[250];
		//��ʼ��ͼ����Ϣ����
		Devpar.ImgHeight = gStructVarible.ImgHeight;
		Devpar.ImgWidth = gStructVarible.ImgWidth;
		Devpar.Threshold = gStructVarible.Threshold;
		Devpar.LengthMin = gStructVarible.LengthMin;
		Devpar.LengthMax = gStructVarible.LengthMax;
		Devpar.AreaMin = gStructVarible.AreaMin;
		Devpar.AreaMax = gStructVarible.AreaMax;
		Devpar.PictureNum = gStructVarible.PictureNum;
		Devpar.RecPadding = gStructVarible.RecPadding;
		//��λ��������
		const cv::Point directions[8] = { { 0, 1 },{ 1,1 },{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1, 1 } };
		//ͼ��ռ����
		unsigned char *ImgHostdata = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum]; //qwt���������BUG
		unsigned char *m_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//��ֵ��ͼ
		unsigned char *n_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//����ͼ
		unsigned char *c_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//����ͼ	
		unsigned char *temp_ptr = new unsigned char[Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum];//��ʱ����ͼ

		RecupdataInitialSuccessFlag = true;

		while (ExtractPointSuccess == false)//����Ӧ�üӼ�����ʹ����ȡ��Χ�п��Խ�����������ѭ����**************************qwt10.26
		{
			if (HostUpdateRec)//�����������������ݸ�����һ�� ������ȡ��Χ��
			{
				vector<RecData>myTempRec;
				memcpy(ImgHostdata, gRecupImgData, sizeof(unsigned char)*Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum);//��������ڴ����򿽱�ͼ��
																															  //��ֵ��
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
				//����
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
				//��ʴ  c_ptr������
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
				//��λ��
				short xmax;
				short xmin;
				short ymax;
				short ymin;
				// ��Ե����  
				int i, j, counts = 0, curr_d = 0;//counts����ѭ������   curr_d�Ƿ������������ID
				short cLength;
				//��ȡ��λ����
				for (i = 1; i < Devpar.ImgHeight*Devpar.PictureNum - 1; i++)
					for (j = 1; j < Devpar.ImgWidth - 1; j++)
					{
						// ��ʼ�㼰��ǰ��  
						cv::Point b_pt = cv::Point(i, j);
						cv::Point c_pt = cv::Point(i, j);
						// �����ǰ��Ϊǰ����  
						if (255 == c_ptr[j + i * Devpar.ImgWidth])
						{
							cLength = 1;
							xmin = xmax = i;
							ymin = ymax = j;
							/*	bool first_t = false;*/
							bool tra_flag = false;//���ñ�־λ
							c_ptr[j + i * Devpar.ImgWidth] = 0;// �ù��ĵ�ֱ�Ӹ�����Ϊ0  
							while (!tra_flag)// ���и��� 
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
									// ���ٵĹ��̣�Ӧ���Ǹ������Ĺ��̣���Ҫ��ͣ�ĸ���������root��  
									c_pt = cv::Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);
									// �߽��ж�  
									if ((c_pt.x > 0) && (c_pt.x < Devpar.ImgHeight*Devpar.PictureNum - 1) &&
										(c_pt.y > 0) && (c_pt.y < Devpar.ImgWidth - 1))
									{
										// ������ڱ�Ե  
										if (255 == c_ptr[c_pt.x*Devpar.ImgWidth + c_pt.y])
										{
											//���°�Χ��
											xmax = xmax > c_pt.x ? xmax : c_pt.x;
											ymax = ymax > c_pt.y ? ymax : c_pt.y;
											xmin = xmin < c_pt.x ? xmin : c_pt.x;
											ymin = ymin < c_pt.y ? ymin : c_pt.y;
											curr_d -= 2;   //���µ�ǰ����  
											c_ptr[c_pt.x*Devpar.ImgWidth + c_pt.y] = 0;
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
									//ɸѡ��λ��
									if (cLength < Devpar.LengthMax && (cLength > Devpar.LengthMin))
									{
										RecData tempRecData;
										int tempcount = 0;
										if (0.7<double(xmax - xmin) / double(ymax - ymin) < 1.5)//��/��
										{
											for (int k = xmin; k <= xmax; k++)//�ж�Height����
											{
												tempcount += temp_ptr[(ymax + ymin) / 2 + k*Devpar.ImgWidth] > 0 ? 1 : 0;
											}
											for (int k = ymin; k <= ymax; k++)//�ж�width����
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
				//������λ�����������ú����߳�����
				gSingleImgRecNum = myTempRec.size() / Devpar.PictureNum;//����ͼ��λ������
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
		//�ͷ��ڴ�
		delete[]ImgHostdata;
		delete[]m_ptr;
		delete[]n_ptr;
		delete[]c_ptr;
		delete[]temp_ptr;
	}
};

/*----------------------------------ʵ�ֲ�ͼѹ�����ܵ���----------------------------------*/
class TC : public Runnable
{
public:
	HardwareInfo param;									//Ӳ������
	unsigned char* my_in;								//�Դ��е�ԭʼλͼ����
	needmemory memory;									//ѹ�����������Դ�
	needdata staticdata;
	static int mTCindex;
	unsigned char* total_malloc;						//ÿһ���������ļ�ռ���ڴ�
	int pix_index;
public:
	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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
	Description:    ������ʼ�����ݽṹ�ͷ����Դ�ռ�ĳ�Ա����
	Calls:          cudaMalloc()��nppiDCTInitAlloc()��cudaMemcpyAsync()��cudaMallocPitch()��
	nppiEncodeHuffmanSpecInitAlloc_JPEG()�����Ƕ���cuda���еĺ���

	Input:          ��
	Output:         ��
	***************************************************************************************************/
	void Initialize()
	{
		//cudaMalloc((void**)&(this->my_in), imgHeight * imgWidth * sizeof(unsigned char) * 3);	//Ϊmy_in�����Դ�ռ�
		cudaMalloc((void**)&(this->my_in), compress_old_Height * compress_old_Width * sizeof(unsigned char) * 3);
		nppiDCTInitAlloc(&(this->memory).pDCTState);											//Ϊmemory.pDCTState�����Դ�ռ�
		cudaMalloc(&(this->staticdata).pdQuantizationTables, 64 * 4);							//staticdata.pdQuantizationTables�����Դ�ռ�

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


		for (int i = 0; i < oFrameHeader.nComponents; ++i)								//����ͼ���С����һЩ������֮����DCT�任��Huffman������Ҫ�õ�
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
		cudaMalloc(&(this->memory).pDScan, (this->memory).nScanSize);														//Ϊmemory.pDScan�����Դ�ռ�
		nppiEncodeHuffmanGetSize((this->staticdata).aDstSize[0], 3, &(this->memory).nTempSize);
		cudaMalloc(&(this->memory).pDJpegEncoderTemp, (this->memory).nTempSize);											//Ϊmemory.pDJpegEncoderTemp�����Դ�ռ�


		for (int j = 0; j < 3; j++) {
			size_t nPitch1;
			cudaMallocPitch(&(this->memory).pDCT[j], &nPitch1, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height);		//Ϊmemory.pDCT�����ڴ�ռ�
			(this->memory).DCTStep[j] = static_cast<Npp32s>(nPitch1);
			cudaMallocPitch(&(this->memory).pDImage[j], &nPitch1, (this->staticdata).aDstSize[j].width, (this->staticdata).aDstSize[j].height);		//Ϊmemory.pDImage�����Դ�ռ�
			(this->memory).DImageStep[j] = static_cast<Npp32s>(nPitch1);
			dataduiqi[j] = nPitch1;

		}
		for (int i = 0; i < 3; ++i)				//��ʼ���Դ��е�staticdata.apDHuffmanDCTable �� staticdata.apDHuffmanACTable
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
	Description:    ���ȵ���jpegNPP���̵�nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW������
	�����Ƕ�memory.pDImage�е���ƬYUV���ݽ���DCT�任���������������������memory.pDCT�У�

	֮�����jpegNPP���̵�nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R������
	�����ǶԾ���DCT�任���ͼ������memory.pDCT���л��������룬������������memory.pDScan�б��棬�ȴ�д����̡�

	Calls:          nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW()��nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R()�����Ƕ���cuda���еĺ���

	Input:          ��
	Output:         ��
	***************************************************************************************************/
	void process()
	{
		for (int i = 0; i < 3; ++i)													//��YCbCr����ͨ����ͼƬ���ݽ���DCT�任
		{
			nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW((this->memory).pDImage[i], (this->memory).DImageStep[i],
				(this->memory).pDCT[i], (this->memory).DCTStep[i],
				(this->staticdata).pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
				(this->staticdata).aDstSize[i],
				(this->memory).pDCTState);
		}

		nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R((this->memory).pDCT, (this->memory).DCTStep,			//���л������������
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
	Description:    ���jpgͼƬд����̵Ĺ�����
	�Ȱ�JFIF��ǩ��aQuantizationTables��׼������oFrameHeader�ṹͷ��
	��׼�����������oScanHeaderɨ��ͷд���ļ���ΪjpgͼƬ���ļ�ͷ��
	֮��memory.pDScan��������������д���ļ������������.jpgͼƬ��


	Calls:          writeMarker()��writeJFIFTag()��writeQuantizationTable()��writeHuffmanTable()
	��Щ����������useful.h��

	Input:          ��
	Output:         ��
	***************************************************************************************************/
	//void writedisk(char* OutputFile)
	void writedisk(int picture_num, Package* a, int bag_index)
	{
		unsigned char *pDstJpeg = new unsigned char[(this->memory).nScanSize];							//Ϊÿһ��.jpgͼƬ���ݿ��ٻ�����
		unsigned char *pDstOutput = pDstJpeg;

		oFrameHeader.nWidth = (this->staticdata).oDstImageSize.width;
		oFrameHeader.nHeight = (this->staticdata).oDstImageSize.height;

		writeMarker(0x0D8, pDstOutput);
		writeJFIFTag(pDstOutput);
		writeQuantizationTable(aQuantizationTables[0], pDstOutput);										//д���׼������
		writeQuantizationTable(aQuantizationTables[1], pDstOutput);
		writeFrameHeader(oFrameHeader, pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);												//д������������
		writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
		writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
		writeScanHeader(oScanHeader, pDstOutput);

		cudaMemcpy(pDstOutput, (this->memory).pDScan, (this->memory).nScanLength, cudaMemcpyDeviceToHost);
		pDstOutput += (this->memory).nScanLength;
		writeMarker(0x0D9, pDstOutput);
		char szOutputFiler[100];
		sprintf_s(szOutputFiler, "%s\\%d.jpg", gStructVarible.ImgSavePath, picture_num);

		memcpy(total_malloc + pix_index, pDstJpeg, static_cast<int>(pDstOutput - pDstJpeg));			//����һ����.jpgͼƬ���ݿ���������ڴ���total_malloc
		pix_index += static_cast<int>(pDstOutput - pDstJpeg);
		//a->Form_one_head(bag_index / gStructVarible.PictureNum, szOutputFiler, pDstOutput - pDstJpeg);

		a->Form_one_head(bag_index / gStructVarible.PictureNum, picture_num, pDstOutput - pDstJpeg);					//���һ��.jpgͼƬ��Ӧ�İ�ͷ�� bag_index / gStructVarible.PictureNum�������ǵڼ���ͼ

																														//{
																														//Write result to file.
																														//std::ofstream outputFile1(OutputFile, ios::out | ios::binary);
																														//outputFile1.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
																														//}

		delete[] pDstJpeg;
	}


	/*************************************************************************************************
	Function:       void memoryfree()
	Description:    ����������ͷ�֮ǰ����õ��Դ�ռ�
	Calls:          cudaFree()��nppiEncodeHuffmanSpecFree_JPEG()��nppiDCTFree()
	���Ƕ���cuda���еĺ���

	Input:          ��
	Output:         ��
	***************************************************************************************************/
	void memoryfree()																	//�ͷ�֮ǰ������ڴ���Դ�
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
	Description:    �Ƕ��߳���T���е���ں���������ѹ��ģ������￪ʼ����
	Calls:          ���ε�����Initialize()��RGBtoYUV <<<blocks, threads >>>��process()��
	writedisk(szOutputFile)��memoryfree()

	Input:          ��
	Output:         ��
	***************************************************************************************************/
	void Run()
	{
		char ImgoutputPath[255];
		total_malloc = new unsigned char[100000000];
		pix_index = 0;
		char szOutputFile[100];
		clock_t start, end;
		int img_index;									//ͼ������
		int mFlagIndex = 0;
		int OutPutInitialIndex = 0;						//�����Bin�ļ���ʼ������
		int Bufferoffset = 0;							//������ƫ����
		bool DatafullFlag = false;		//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�

		cudaSetDevice((this->param).GpuId);
		this->Initialize();

		cout << "T GPU ��" << param.GpuId << " initial success!" << endl;

		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			img_index = 0;								//ͼ�����
			Bufferoffset = 0;
			//��ȡ����
			while (true)
			{
				gComressReadDataLock.lock();
				mTCindex = mTCindex % (HardwareParam.DeviceCount + 1);
				if (gComressionBufferEmpty[mTCindex] == false && gComressionBufferWorking[mTCindex] == false)
				{
					//��ҳ���ڴ��־λ��Ϊ����״̬--���а�
					gComressionBufferWorking[mTCindex] = true;
					OutPutInitialIndex = gComressionBufferStartIndex[mTCindex] * Bufferlength;//��ȡͼ��������
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

			//ѹ��pImgͼƬ
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
					data_bag.file.open(data_bag.Fname, ios::out | ios::binary);										//50��.jpg��ɺ󣬴�һ���������ļ�
																													//data_bag.Form_total_head();																		//�������50��ͼƬ�İ�ͷ��Ϣ
					data_bag.Form_total_head(compress_imgWidth, compress_imgHeight, gStructVarible.PictureNum, OutPutInitialIndex);
					data_bag.file.write(data_bag.head_cache, data_bag.head_bias);									//д�����а�ͷ
					data_bag.file.write(reinterpret_cast<const char *>(total_malloc), static_cast<int>(pix_index)); //д����������
					data_bag.file.close();
					//data_bag.UnPack(data_bag.Fname);
					compress_write_lock.unlock();
					memset(total_malloc, 0, 100000000);																//���������
					pix_index = 0;																					//��������������
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
				img_index = img_index + gStructVarible.PictureNum;                                                //picture_index��һ��ʵ���ܵ�ͼƬ���
			}
		}
		delete[] total_malloc;
		this->memoryfree();
	}
};
int TC::mTCindex = 0;

//----------------------------------ʵ�ֻҶ�ͼѹ�����ܵ���------------------------------------------//
//�˺����Ա��
class T : public Runnable									//���ڸ����ʵ�ֺ�ǰ��TC�༰�����ƣ����Բ��ٽ���ע��
{
public:
	HardwareInfo param;										//Ӳ������
	gpuneedmemory memory[GRAYCompressStreams];
	static int mTindex;
	needconstdata staticdata;                               //ѹ���������õ��ĳ�������
	RIM   ImageSize;                                        //��¼ͼ���С����ÿ��ͼ����������Ķ�����
	size_t Org_Pitch;                                       //��¼ԭʼͼ�����ݶ����Ŀ��
	int  h_MCUtotal;                                        //����ͼ�����8*8���ؿ�����
	cudaStream_t stream[GRAYCompressStreams];               //����CUDA��
	int   stridef;
	cpuneedmemory cpumemory[GRAYCompressStreams];           //���ѹ��ͼ�������CPU�ϵ�ԭʼͼ��λͼ���ݺ����ձ���ͼ������
	unsigned char* total_malloc;                            //����������ڴ��еĻ���ռ�
	int pix_index;
public:
	void mydelay(double sec)                               //��ʱ����������ͼ�����ݻ������ĸ���
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
	//*Description:    ������ʼ�����ݽṹ�ͷ����Դ�ռ�ĳ�Ա����
	//*Calls:          cudaMalloc()��cudaMemcpyAsync()��cudaMallocPitch()���Ƕ���cuda���еĺ���
	//*Input:          ImageSize ��¼ͼ�����ݴ�С�����ڷ����Դ����ڴ�
	//*Output:         ��
	//***************************************************************************************************
	void Initialize() {
		size_t nPitch;
		this->stridef = ALIGN(compress_old_Width, 4);
		(this->ImageSize).width = ALIGN(compress_old_Width, 8);
		(this->ImageSize).height = ALIGN(compress_old_Height, 8);
		int h_MCUtotal = (this->ImageSize).height*(this->ImageSize).width / 64;
		int ARRAY_SIZE = ALIGN(h_MCUtotal + 1025, 1024);
		int ARRAY_SIZE1 = ALIGN(h_MCUtotal / 1024 + 1025, 1024);

		//Ϊ�������ͼ������ȷ���ڴ��С
		(this->staticdata).nScanSize = (this->ImageSize).width * (this->ImageSize).height * 2;
		(this->staticdata).nScanSize = (this->staticdata).nScanSize > (10 << 20) ? (this->staticdata).nScanSize : (10 << 20);

		for (int i = 0; i < GRAYCompressStreams; i++) {
			//Ϊÿһ���������Դ����ڴ�
			cudaMallocPitch((void **)&(this->memory[i].d_bsrc), &(this->ImageSize.StrideF), (this->ImageSize).width * sizeof(BYTE), (this->ImageSize).height);      //Ϊmy_in�����Դ�ռ�
			cudaMallocPitch((void **)&(this->memory[i].d_ydst), &nPitch, (this->ImageSize).width * (this->ImageSize).height * sizeof(BSI16), 1);
			cudaMallocPitch((void **)&(this->memory[i].d_JPEGdata), &nPitch, (this->ImageSize).width * sizeof(BYTE)*(this->ImageSize).height, 1);
			cudaMalloc((void **)&(this->memory[i].last_JPEGdata), (10 << 20));
			cudaMalloc((void **)&(this->memory[i].prefix_num), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].last_prefix_num), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].dc_component), ARRAY_SIZE * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].d_blocksum), 768 * sizeof(int));
			cudaMalloc((void **)&(this->memory[i].d_datalen), sizeof(int));
			//����CUDA��
			cudaStreamCreate(&(this->stream[i]));
			//����CPU�ڴ�
			//cudaHostAlloc((BYTE**)&(this->cpumemory[i]).pDstJpeg, (this->staticdata).nScanSize, cudaHostAllocDefault);    //���ձ�������
			(this->cpumemory[i]).pDstJpeg = new unsigned char[(this->staticdata).nScanSize];
			this->cpumemory[i].pDstOutput = this->cpumemory[i].pDstJpeg;
		}

		//-------------------------Ϊ�Ҷ�ͼ��ѹ�����ó�������--------------------
		cudaMalloc(&(this->staticdata).DEV_STD_QUANT_TAB_LUMIN, 64 * sizeof(float));
		cudaMalloc(&(this->staticdata).DEV_ZIGZAG, 64 * sizeof(int));
		{
			//--------------------��������������--------------------------------
			float temp[64];
			for (int i = 0; i<64; i++) {
				temp[i] = 1.0f / (float)STD_QUANT_TAB_LUMIN[i] * C_norm * C_norm;
			}
			cudaMemcpyAsync((this->staticdata).DEV_STD_QUANT_TAB_LUMIN, temp, 64 * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMemcpyAsync((this->staticdata).DEV_ZIGZAG, aZIGZAG, 64 * sizeof(float), cudaMemcpyHostToDevice);
		{
			//----------------��ʼ��huffman��
			GPUjpeg_huffman_encoder_value_init_kernel << <32, 256 >> >();  // 8192 threads total
																		   // ����GPU�汾��Huffman�� ( CC >= 2.0)
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
	//**Description:    ����ѹ��ͼ��ĺ���
	//**Input:          Size ��¼ͼ�����ݴ�С�����ڷ����Դ����ڴ�
	//**Output:         ��
	//***************************************************************************************************
	void process() {
		const int ARRAY_SIZE = ImageSize.width * ImageSize.height;
		const int  h_MCUtotal = ARRAY_SIZE / 64;                               //ͼ�������ܵ�8*8MCU��Ԫ

		const int Code_blocks = (h_MCUtotal + CODE_THREADS - 1) / CODE_THREADS;
		int Blocksums;
		int prexsum_blocks = 1;
		int prexsum_threads = (h_MCUtotal - 1) / CODE_THREADS;

		//prefix_sumǰ׺����̷߳���
		int preSum_Blocks = (h_MCUtotal + 1023) / 1024;
		//DCT�̷߳���
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

			//����ÿ��mcu�������ľ���λ�ã�ǰ׺����㷨
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

			//��ͼ�����ݽ��б��봦��
			data_shift_kernel << <Code_blocks, CODE_THREADS, 0, this->stream[i] >> >(this->memory[i].d_JPEGdata,
				this->memory[i].prefix_num, h_MCUtotal, this->memory[i].d_datalen,
				this->memory[i].dc_component, this->memory[i].last_prefix_num);
			//����ÿ��MCU BYTE���ľ���λ�ã�ǰ׺����㷨
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

			//�õ�ͼ�����ݱ��볤��
			cudaMemcpyAsync(&this->cpumemory[i].dst_JPEGdatalength, (this->memory[i]).d_datalen, sizeof(int), cudaMemcpyDeviceToHost, this->stream[i]);
		}
	}

	void writedisk(int picture_num, Package* a, int bag_index)
	{
		for (int i = 0; i < GRAYCompressStreams; i++) {
			this->cpumemory[i].pDstOutput = this->cpumemory[i].pDstJpegDataStart;
			//�����ͼ�����ݴ���
			cudaMemcpyAsync(this->cpumemory[i].pDstOutput, this->memory[i].last_JPEGdata,
				this->cpumemory[i].dst_JPEGdatalength,
				cudaMemcpyDeviceToHost, this->stream[i]);
			//-------------�ȴ�Stream��ִ�����
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
			unsigned short len = 2 + 1 + 2 + 2 + 1 + 3 * 3;   //3����ɫ������
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
			// output DHT AC   0xC4     ������(Huffman)�� 
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

			// output DHT DC 0xC4    ������(Huffman)��
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

			// output SOS  0xDA�� ɨ���߿�ʼ
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
	//�ͷ�������Դ����ڴ�
	void memoryfree()
	{
		for (int i = 0; i < GRAYCompressStreams; i++) {
			//�ͷ��Դ�
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
		int img_index;//ͼ������
		int cudaStreams_imgindex = 0; //ÿ������ͼ������
		int mFlagIndex = 0;
		int OutPutInitialIndex = 0; //�����Bin�ļ���ʼ������
		int Bufferoffset = 0;       //������ƫ����
		bool DatafullFlag = false;//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�
								  //���Զ���ͼƬ�Ƿ�ɹ�-------------------------------------------------------------------------------------
		cv::Mat img1(5120, 5120, CV_8UC1);
		cudaSetDevice((this->param).GpuId);
		this->Initialize();
		cout << "T GPU ��" << param.GpuId << " initial success!" << endl;
		WriteJpgheader();
		while (!ExtractPointSuccess)
		{
			mydelay(0.01);
			img_index = 0;//ͼ�����
			Bufferoffset = 0;
			//������
			while (true)//������Ҫ�ģ��������ú����һ��
			{
				gComressReadDataLock.lock();
				mTindex = mTindex % (HardwareParam.DeviceCount + 1);
				if (gComressionBufferEmpty[mTindex] == false && gComressionBufferWorking[mTindex] == false)
				{
					//��ҳ���ڴ��־λ��Ϊ����״̬--���а�
					gComressionBufferWorking[mTindex] = true;
					OutPutInitialIndex = gComressionBufferStartIndex[mTindex] * Bufferlength;//��ȡͼ��������
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

			//ѹ��pImgͼƬ
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

					//д����
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
					//cout << "T GPU ��" << param.GpuId << " Index" << OutPutInitialIndex << " ����" << double(end - start) / CLOCKS_PER_SEC <<"  ��ʱ�䣺"<< double(end2 - start) / CLOCKS_PER_SEC<< endl;
					break;
				}
				int picture_index = OutPutInitialIndex + img_index;
				//Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.PictureNum;
				Bufferoffset = gStructVarible.ImgWidth * gStructVarible.ImgHeight * img_index;
				//��ͼ�����ݴ��䵽GPU
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
			//-------------����CUDA��
			cudaStreamDestroy(this->stream[i]);
	}
};
/*----------------------------------ʵ�ֻҶ�ͼѹ�����ܵ���--------------------------------*/
//npp����ð�
//class T : public Runnable									//���ڸ����ʵ�ֺ�ǰ��TC�༰�����ƣ����Բ��ٽ���ע��
//{
//public:
//	HardwareInfo param;										//Ӳ������
//	needmemory memory;
//	needdata staticdata;
//	static int mTindex;
//	static int test_number;
//	unsigned char* total_malloc;
//	int pix_index;
//
//
//public:
//	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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
//			//NPP_CHECK_CUDA(cudaMallocPitch(&myImage1[j], &nPitch1, aSrcSize[j].width, aSrcSize[j].height));   ԭ��
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
//		int img_index;//ͼ������
//		int mFlagIndex = 0;
//		int OutPutInitialIndex = 0; //�����Bin�ļ���ʼ������
//		int Bufferoffset = 0;//������ƫ����
//		bool DatafullFlag = false;//��־λ����Ϊtrue��ʱ�򣬱�ʾ��GPU��Ӧ�������������У�������һ������Ч���ݡ�
//								  //���Զ���ͼƬ�Ƿ�ɹ�------------------------------------------------------------------------------------------------------------
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
//		cout << "T GPU ��" << param.GpuId << " initial success!" << endl;
//
//		while (!ExtractPointSuccess)
//		{
//			mydelay(0.01);
//			img_index = 0;//ͼ�����
//			Bufferoffset = 0;
//			//������
//			while (true)//������Ҫ�ģ��������ú����һ��
//			{
//				gComressReadDataLock.lock();
//				mTindex = mTindex % (HardwareParam.DeviceCount + 1);
//				if (gComressionBufferEmpty[mTindex] == false && gComressionBufferWorking[mTindex] == false)
//				{
//					//��ҳ���ڴ��־λ��Ϊ����״̬--���а�
//					gComressionBufferWorking[mTindex] = true;
//					OutPutInitialIndex = gComressionBufferStartIndex[mTindex] * Bufferlength;//��ȡͼ��������
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
//			//ѹ��pImgͼƬ
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
//					//д����
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
//					cout << "T GPU ��" << param.GpuId << " Index" << OutPutInitialIndex << " ����" << double(end - start) / CLOCKS_PER_SEC << "  ��ʱ�䣺" << double(end2 - start) / CLOCKS_PER_SEC << endl;
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
/*----------------------------------���ݸ�����--------------------------------------------*/
class  ReadImg : public Runnable
{
public:
	bool ExtractPointWorkingFlag = false;//��ʾ ����ڹ���
	bool CompressionWorkingFlag = false;//��ʾ  ѹ���ڹ��� 
	Parameter Devpar;//��������	
	~ReadImg()
	{
	}
	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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
		//��ʼ����־λ
		for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
		{
			//ҳ��
			PageLockBufferEmpty[i] = true;
			PageLockBufferWorking[i] = false;
			PageLockBufferStartIndex[i] = 0;
			//ѹ��
			gComressionBufferEmpty[i] = true;
			gComressionBufferWorking[i] = false;
			gComressionBufferStartIndex[i] = 0;
		}
		//cout << "ReadImg initial success!" << endl;
		while (!ExtractPointSuccess) // ʵ������ı�־λ
		{
			mydelay(0.01);
			for (int i = 0; i <HardwareParam.DeviceCount * 2; i++)//������ڱ��������Ӧbuffer�ı�־λ
			{
				ExtractCopySuccess = false;
				ComressionCopySuccess = false;
				if (CameraBufferFull[i]) //�����Ӧ���ڴ��Ƿ��п�������--��Ϊtrueʱ,���ʾ�����Ӧ�ڴ��i��Buffer�п�������
				{
					//������λ����������λ�л�����
					if (gStructVarible.RecModelFlag == true && HostUpdateRec == false)
					{
						memcpy(gRecupImgData, gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgChannelNum*Devpar.ImgWidth* Devpar.ImgHeight*Devpar.PictureNum);//��������ڴ����򿽱�ͼ��
						HostUpdateRec = true;
					}

					//������ڴ濽�����ݵ�ҳ���ڴ�
					if (ExtractPointWorkingFlag)
					{
						while (1) //����ҳ��Buffer,�ж�ҳ���ڴ滺�����Ƿ��п���Buffer��
						{
							mPageLockBufferIndex = mPageLockBufferIndex % (HardwareParam.DeviceCount + 1);
							if (PageLockBufferEmpty[mPageLockBufferIndex])//��ĳһҳ��Ϊ�� 
							{
								memcpy(gHostBuffer[mPageLockBufferIndex], gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgHeight*Devpar.ImgWidth *Devpar.ImgChannelNum* Bufferlength);//�������ݵ�ҳ��
								ExtractCopySuccess = true;
								PageLockBufferEmpty[mPageLockBufferIndex] = false;// ����������֮�� ��־λ��Ϊfalse;
								PageLockBufferStartIndex[mPageLockBufferIndex] = BufferBlockIndex[i];//��������
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

					// ������ڴ濽����ѹ��������
					if (CompressionWorkingFlag)
					{
						while (1) //����ҳ��Buffer,�ж�ҳ���ڴ滺�����Ƿ��п���Buffer��
						{
							mCompressionBufferindex = mCompressionBufferindex % (HardwareParam.DeviceCount + 1);
							if (gComressionBufferEmpty[mCompressionBufferindex])//��ĳһҳ��Ϊ�� 
							{
								memcpy(gHostComressiongBuffer[mCompressionBufferindex], gCameraBuffer[i], sizeof(unsigned char)*Devpar.ImgHeight*Devpar.ImgWidth *Devpar.ImgChannelNum* Bufferlength);//�������ݵ�ҳ��
								ComressionCopySuccess = true;
								gComressionBufferEmpty[mCompressionBufferindex] = false;// ����������֮�� ��־λ��Ϊfalse;
								gComressionBufferStartIndex[mCompressionBufferindex] = BufferBlockIndex[i];//��������
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

					//����ڴ��Ӧ��־λ��false
					if (ExtractCopySuccess&&ComressionCopySuccess)
						CameraBufferFull[i] = false;
				}
			}
		}

	}
};

/*----------------------------------ģ�����ݲ�����----------------------------------------*/
class  DataRefresh : public Runnable
{
public:
	Parameter Devpar;//��������	
	~DataRefresh()
	{
	}
	void mydelay(double sec)//��ʱ����������ͼ�����ݻ������ĸ���
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
		//��ʼ������
		Devpar.ImgHeight = gStructVarible.ImgHeight;
		Devpar.ImgWidth = gStructVarible.ImgWidth;
		Devpar.PictureNum = gStructVarible.PictureNum;
		Devpar.ImgChannelNum = gStructVarible.ImgChannelNum;
		clock_t start, end;
		char  path[250];
		//��ͼ��
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
		//��ʼ����������
		for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
		{
			BufferBlockIndex[i] = i - HardwareParam.DeviceCount * 2;
			CameraBufferFull[i] = false;
		}
		//cout << "DataRefresh initial success!" << endl;/*������*/
		mydelay(2);
		//cout << " start!" << endl;
		//ģ�������������
		start = clock();
		for (int q = 0; q <3; q++)   //5
		{
			for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
			{

				BufferBlockIndex[i] += HardwareParam.DeviceCount * 2;
				//������ٶȹ�������������ӡ;
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
		//cout << "ʵ��:50��ͼƬ����ʱ��=" << Timedatarefresh << "  over" << endl;
		delete[] Img1;

	}
};


//--------------------------------------------����------------------------------------------//

/*����ӿں���*/

//���ö�̬���һ��
//��Ӳ���豸��ʼ��������Ӧ����
/*************************************************
��������: GetDiskSpaceInfo  //

��������: �������ڻ�ȡ����·������λ�ô���ʣ������(GB)�� //

���������LPCWSTR pszDrive ��·������λ���̷���
.		  ����"D:\1.bmp"��Ϊ"D:\"��//

����������գ�//

����ֵ  : RemainingSpace(int��) -- ʣ���������(GB)//

����˵��: ����ֻ����Ӳ����ʼ��ʱ����һ�Σ�
.		  �������������������������ȡ����
.		  ���������пռ䡢�������ȴ�����Ϣ
.		  �ȣ�δ�����ӿ�//

*************************************************/
int GetDiskSpaceInfo(LPCWSTR pszDrive)
{
	DWORD64 qwFreeBytesToCaller, qwTotalBytes, qwFreeBytes;
	DWORD dwSectPerClust, dwBytesPerSect, dwFreeClusters, dwTotalClusters;
	BOOL bResult;

	//ʹ��GetDiskFreeSpaceEx��ȡ������Ϣ����ӡ���  
	bResult = GetDiskFreeSpaceEx(pszDrive,
		(PULARGE_INTEGER)&qwFreeBytesToCaller,
		(PULARGE_INTEGER)&qwTotalBytes,
		(PULARGE_INTEGER)&qwFreeBytes);

	//ʹ��GetDiskFreeSpace��ȡ������Ϣ����ӡ���  
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
��������: HardwareInit  //

��������: Ӳ����ʼ���� //

���������null//

���������HardwareInfo *HardwareProp �� Ӳ��������Ϣ��//

����ֵ  : (int��) -- ��ʼ���ɹ���ʧ�ܱ�־//

����˵��: ���������������ʼ��ʱ����
.		  ��ϵͳʹ��Ӳ����Դ���ã�//

*************************************************/
IMGSIMULATION_API int HardwareInit(HardwareInfo *HardwareProp)
{
	if (gWorkingGpuId.size() != 0)
		gWorkingGpuId.clear();
	cudaGetDeviceCount(&gDeviceCount);

	//������Ϣ���ڽṹ��HardwareParam��
	HardwareParam.DeviceCount = 0;//GPU�豸������
	HardwareParam.DiskRemainingSpace = GetDiskSpaceInfo(L"C:/pic");//C��ʣ��ռ�
	if (HardwareParam.DiskRemainingSpace < DiskRemainingSpaceThreshold)//%%%��ʱ��Ϊ100G%%%
	{
		return 1;//���̴洢�ռ䲻��
	}
	for (int i = 0; i<gDeviceCount-1; i++)
	{
		cudaDeviceProp DevProp;
		cudaGetDeviceProperties(&DevProp, i);
		HardwareProp->major = DevProp.major;
		HardwareProp->minor = DevProp.minor;
		if (DevProp.major > 5)//������������5ʱ
		{
			gWorkingGpuId.push_back(i);
		}
	}
	if (HardwareParam.DeviceCount > 5 && HardwareParam.DeviceCount < 1)
	{
		return 2;//����ͬʱ֧��5��GPU
	}
	HardwareParam.DeviceCount = gWorkingGpuId.size();//GPU�豸��Ŀ
	HardwareProp->DeviceCount = HardwareParam.DeviceCount;
	HardwareParam.ExPointThreads = HardwareParam.DeviceCount;//����߳���
	HardwareProp->ExPointThreads = HardwareParam.DeviceCount;
	HardwareParam.CompThreads = HardwareParam.DeviceCount;//ѹ���߳���
	HardwareProp->CompThreads = HardwareParam.DeviceCount;

	return 0;
}
//-------------------------------------------------------����----------------------------------------//

/*************************************************
��������: Image_Pretreatment  //

��������: ��������ͼ��Ԥ���� //

���������const char *path ��ͼ���ļ���·����
.		  const char *exten �� ͼ���ʽ������".bmp"��
.		  int ChooseMode : Ԥ����ѡ��--1 ͼ������ڴ�
.									 --2 �ڴ��ͷ�//

���������gHostImage[i]�������ͼ�����ݵ����飻//

����ֵ  : gHostPathImgNumber(int��) -- ·����ͼ������//

����˵��: �����ڵ���ʱʹ�ã�����ͼ���������//

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
		//ͼ��Ԥ������Ӳ�����������ڴ�
#ifdef Pretreatment
		char strFilename[100];
		int mWidth;
		int mHeight;
		for (int i = 0; i < gHostPathImgNumber; i++)
		{
			sprintf_s(strFilename, "%s\\%d.bmp", path, i + 1); //��ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
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
		//�����ڴ��ͷ�
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
��������: SimulationImageTest  //

��������: ����ԭͼ������ԣ� //

���������const char *path ������ͼ��·����//

���������Infomation *Info �� ����ʵ�����ݣ�//

����ֵ  : bool -- ʵ��ɹ���־λ//

����˵��: ������������������ͼ�ڲ�ͬʵ��ģʽ��
.		  ʵ�����ܵĲ��ԣ�
.		  ���԰������������ԡ���ѹ�����ԡ����ѹ������//

*************************************************/
IMGSIMULATION_API bool SimulationImageTest(const char *path, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	gHostPathImgNumber = 5;//����ͼƬ��������
	Info->ImgProcessingNumbers = gHostPathImgNumber;
	/****  ͼƬ����  ****/
	for (int i = 0; i < gHostPathImgNumber; i++)//ΪͼƬ������ҳ�ڴ�
	{
		err = cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth  *gStructVarible.PictureNum * sizeof(unsigned char), cudaHostAllocDefault);
		if (gStructVarible.ImgBitDeep == 24)
		{
			err = cudaHostAlloc((void**)&gHostColorImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth  *gStructVarible.PictureNum * 3 * sizeof(unsigned char), cudaHostAllocDefault);
		}
	}
	int Picoffset = gStructVarible.ImgHeight * gStructVarible.ImgWidth;//���ŻҶ�ͼƬ��ַƫ����
	int PicoffsetColor = gStructVarible.ImgHeight * gStructVarible.ImgWidth * 3;//����ͼƬ��ַƫ����
	for (int i = 0; i < gHostPathImgNumber; i++)//��ȡͼƬ
	{
		for (int j = 0; j < gStructVarible.PictureNum; j++)
		{
			RmwRead8BitBmpFile2Img(path, gHostColorImage[i] + j * PicoffsetColor, gHostImage[i] + j * Picoffset, &mWidth, &mHeight);
		}
	}
	if (gStructVarible.RecModelFlag == 1)
		GetImgBoxHost(path);//��ȡ��Χ��
	Info->DeviceCount = HardwareParam.DeviceCount;
	Info->CPUThreadCount = ExtractPointThreads;
	clock_t start, finish;
	float Difftime;//ʱ���
	float ImageSize;//ͼ��ߴ�
	int ImgChannel;//ͼ��ͨ��
	int ThreadID;

	/****  �������� ****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	//����߳���ΪGPU�豸��
	pExecutor->Init(1, HardwareParam.ExPointThreads, 1);
	SIM *ExtractPoint = new SIM[HardwareParam.ExPointThreads];
	RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
	//RecS recs;

	if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
	{
		start = clock(); //��ʱ��ʼ
		ThreadID = 0x01;//�̺߳�
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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

			/**** ��ȡ��־����� ****/
			pExecutor->Execute(&ExtractPoint[i], ThreadID);
			ThreadID = ThreadID << 1;
		}

		pExecutor->Terminate();//��ֹ�߳�
		delete pExecutor;//ɾ���̳߳�	
		finish = clock();//��ʱ����					 
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;//�õ����μ�¼֮���ʱ���
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	else //����ģʽ
	{
		start = clock(); //��ʱ��ʼ
		ThreadID = 0x01;//�̺߳�
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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

			/**** ��ȡ��־����� ****/
			pExecutor->Execute(&RecExtractPoint[i], ThreadID);
			ThreadID = ThreadID << 1;
		}

		pExecutor->Terminate();
		delete pExecutor;
		finish = clock();//��ʱ����
						 //�õ����μ�¼֮���ʱ���
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	///****  ��ѹ������****/
	//CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
	//pExecutor1->Init(1, HardwareParam.CompThreads, 1);
	//T *Compression_grey = new T[HardwareParam.CompThreads];
	//TC *Compression = new TC[HardwareParam.CompThreads];

	//start = clock(); //��ʱ��ʼ
	//ThreadID = 0x01;//�̺߳�����
	//for (int i = 0; i < HardwareParam.ExPointThreads; i++)
	//{
	//	/**** �������� ****/
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
	//finish = clock();//��ʱ����
	//				 //�õ����μ�¼֮���ʱ���
	//Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
	//Info->CompressionTimes = Difftime;
	//ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
	//ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
	//Info->CompressionSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;

	///****  �����ѹ��ͬ������****/
	//CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
	//pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads, 1);

	//start = clock(); //��ʱ��ʼ
	//ThreadID = 0x01;//�̺߳�����
	//if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
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
	//finish = clock();//��ʱ����
	//				 //�õ����μ�¼֮���ʱ���
	//Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
	//Info->SynchronizeTimes = Difftime;
	//ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
	//ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
	//Info->SynchronizeSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	//�ͷ��ڴ�
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
	//������ѹ��
	Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//����ͼƬ��С
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/sʱ������ˢ��ʱ�䡣
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
		//ÿ��ʵ��֮����ʱ����
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


/****************************************����ʵ�����********************************************/
//qwe �����ܺ���
IMGSIMULATION_API bool SimulationExperient(int ChooseMode)
{
	clock_t start, finish;
	Infomation *Info;
	float Difftime;//ʱ���
	float ImageSize;//ͼ��ߴ�
	int ImgChannel;//ͼ��ͨ��
	int ThreadID;

	//cout << "�豸��Ŀ:" << HardwareParam.DeviceCount << endl;

	switch (ChooseMode)
	{
	case 1://�����
	{
		/****  ��������****/
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
		if (gStructVarible.RecModelFlag == false)//ȫͼģʽ
		{
			ThreadID = 0x01;//�̺߳�
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();//��ֹ�߳�
		}
		else //����ģʽ
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			ThreadID = 0x01;//�̺߳�
							/**** ��ȡ��־����� ****/
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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
				/**** ��ȡ��־����� ****/
				pExecutor->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
			pExecutor->Execute(&recupdate, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();

			cout << "ʵ�����" << endl;
			delete pExecutor;
		}
		break;
	}
	case 2://��ѹ��
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, 10, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		DataRefresh  datarefresh;
		ReadImg  readimg;
		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = false;
		ThreadID = 0x01;//�̺߳�����
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
	case 3://���&ѹ��
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
		ThreadID = 0x01;//�̺߳�
		//����߳�
		if (gStructVarible.RecModelFlag == false)//ȫͼģʽ
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //����ģʽ
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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
				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
				pExecutor2->Execute(&recupdate, ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//ѹ���߳�
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
		//��������+��ͼ�߳�
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

//qwe ����ѹ��ͬ��
IMGSIMULATION_API void  SimulationTestSynchronize(const char *path, Infomation *Info)
{

	//������ѹ��
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//����ͼƬ��С
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/sʱ������ˢ��ʱ�䡣
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

		//ÿ��ʵ��֮����ʱ����
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

//qwe �����
IMGSIMULATION_API void  SimulationTestExtractPoint(const char *path, Infomation *Info)
{

	//������ѹ��
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//����ͼƬ��С
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/sʱ������ˢ��ʱ�䡣
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
		//ÿ��ʵ��֮����ʱ����
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

//qwe ��ѹ��
IMGSIMULATION_API void  SimulationTestComression(const char *path, Infomation *Info)
{

	//������ѹ��
	//Bufferlength = 50;
	Memory_application();
	Timedatarefresh = 1;
	double  SiglePicSize = double(gStructVarible.ImgHeight*gStructVarible.ImgWidth) / (1024 * 1024);//����ͼƬ��С
	double minTimeRefresh = Bufferlength*SiglePicSize / (2 * 1024);//2G/sʱ������ˢ��ʱ�䡣
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
		//ÿ��ʵ��֮����ʱ����
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
��������: OnlineImageExperiment  //

��������: ����ʵ��ģ��--ȫͼģʽ�� //

���������const char *Imgpath ������ʵ��ͼ��·����
.		  ChooseMode ��1 �����
.					   2 ��ѹ��
.					   3 ���&ѹ��//

���������Infomation *Info �� ����ʵ�����ݣ�//

����ֵ  : bool -- ʵ��ɹ���־λ//

����˵��: ����ѡ���ԵĽ�������ģʽ������ʵ��
.		  ������ģʽͨ���������ò���ѡ��//

*************************************************/
IMGSIMULATION_API bool OnlineImageExperiment(int ChooseMode, const char *Imgpath, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	clock_t start, finish;
	float Difftime;//ʱ���
	float ImageSize;//ͼ��ߴ�
	int ImgChannel;//ͼ��ͨ��
	int ThreadID;

	switch (ChooseMode)
	{
	case 1://�����
	{
		/****  ��������****/
		CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
		pExecutor->Init(1, HardwareParam.ExPointThreads, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
		{
			start = clock(); //��ʱ��ʼ
			ThreadID = 0x01;//�̺߳�
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Terminate();//��ֹ�߳�
			delete pExecutor;//ɾ���̳߳�	
			finish = clock();//��ʱ����
							 //�õ����μ�¼֮���ʱ���
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		else //����ģʽ
		{
			start = clock(); //��ʱ��ʼ
			ThreadID = 0x01;//�̺߳�
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Terminate();
			delete pExecutor;
			finish = clock();//��ʱ����
							 //�õ����μ�¼֮���ʱ���
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		break;
	}
	case 2://��ѹ��
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, HardwareParam.CompThreads, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];

		start = clock(); //��ʱ��ʼ
		ThreadID = 0x01;//�̺߳�����
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
		finish = clock();//��ʱ����
						 //�õ����μ�¼֮���ʱ���
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->CompressionTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->CompressionSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		break;
	}
	case 3://���&ѹ��
	{
		CThreadPoolExecutor * pExecutor2 = new CThreadPoolExecutor();
		pExecutor2->Init(1, HardwareParam.ExPointThreads + HardwareParam.CompThreads, 1);
		R *ExtractPoint = new R[HardwareParam.ExPointThreads];
		RecR *RecExtractPoint = new RecR[HardwareParam.ExPointThreads];
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];

		ThreadID = 0x01;//�̺߳�
		start = clock(); //��ʱ��ʼ
		if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //����ģʽ
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
		finish = clock();//��ʱ����
						 //�õ����μ�¼֮���ʱ���
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->SynchronizeTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->SynchronizeSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		break;
	}
	default: return 1;
	}
	return 0;
}

/*************************************************
��������: OnlineImageExperiment  //

��������: ����ʵ��ģ��--����ģʽ�� //

���������const char *Imgpath ������ʵ��ͼ��·����
.		  ChooseMode ��1 �����
.					   2 ��ѹ��
.					   3 ���&ѹ��//

���������Infomation *Info �� ����ʵ�����ݣ�//

����ֵ  : bool -- ʵ��ɹ���־λ//

����˵��: ����ѡ���ԵĽ�������ģʽ������ʵ��
.		  ������ģʽͨ���������ò���ѡ��//

*************************************************/
IMGSIMULATION_API bool OnlineImageRecExperiment(int ChooseMode, Infomation *Info)
{
	clock_t start, finish;
	int mWidth, mHeight;
	float Difftime;//ʱ���
	float ImageSize;//ͼ��ߴ�
	int ImgChannel;//ͼ��ͨ��
	int ThreadID;

	switch (ChooseMode)
	{
	case 1://�����
	{
		/****  ��������****/
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

		if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
		{
			start = clock(); //��ʱ��ʼ
			ThreadID = 0x01;//�̺߳�
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

				/**** ��ȡ��־����� ****/
				pExecutor->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}

			pExecutor->Execute(&readimg, ThreadID);
			ThreadID = ThreadID << 1;
			pExecutor->Execute(&datarefresh, ThreadID);
			pExecutor->Terminate();//��ֹ�߳�
			delete pExecutor;//ɾ���̳߳�	
			finish = clock();//��ʱ����
							 //�õ����μ�¼֮���ʱ���
			Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
			Info->PointNumbers = SignPoint.PointNumbers;
			Info->ExtractPointTimes = Difftime;
			ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
			ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
			Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
		}
		else //����ģʽ
		{
			ThreadID = 0x01;//�̺߳�
			GetImgBoxHost(gStructVarible.ImgReadPath);
			/**** ��ȡ��־����� ****/
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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

			//cout << "ʵ�����" << endl;
			delete pExecutor;
		}
		break;
	}
	case 2://��ѹ��
	{
		CThreadPoolExecutor * pExecutor1 = new CThreadPoolExecutor();
		pExecutor1->Init(1, HardwareParam.CompThreads + 2, 1);
		T *Compression_grey = new T[HardwareParam.CompThreads];
		TC *Compression = new TC[HardwareParam.CompThreads];
		ReadImg  readimg;
		DataRefresh  datarefresh;

		readimg.CompressionWorkingFlag = true;
		readimg.ExtractPointWorkingFlag = false;

		ThreadID = 0x01;//�̺߳�����
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
	case 3://���&ѹ��
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

		ThreadID = 0x01;//�̺߳�
		if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
		{
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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
				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&ExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		else //����ģʽ
		{
			GetImgBoxHost(gStructVarible.ImgReadPath);
			for (int i = 0; i < HardwareParam.ExPointThreads; i++)
			{
				/**** �������� ****/
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
				/**** ��ȡ��־����� ****/
				pExecutor2->Execute(&RecExtractPoint[i], ThreadID);
				ThreadID = ThreadID << 1;
				pExecutor2->Execute(&recupdate, ThreadID);
				ThreadID = ThreadID << 1;
			}
		}
		//ѹ���߳�
		for (int i = 0; i < HardwareParam.ExPointThreads; i++)
		{
			/**** �������� ****/
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
		//��������+��ͼ�߳�
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
��������: OnlineImageRefresh  //

��������: ����ʵ���ȡ������ͼ�� //

�����������//

�����������//

����ֵ  : ��//

����˵��: //

*************************************************/
IMGSIMULATION_API int OnlineImageRefresh(unsigned char *pImg)
{
	if (gCameraBuffer[0] == NULL)
		return 1;
	//pImgָ���ڴ��ڽ�������룬��СΪ����ͼ���С
	memcpy(pImg, gCameraBuffer[0], gStructVarible.ImgWidth * gStructVarible.ImgHeight * gStructVarible.ImgChannelNum * sizeof(unsigned char));
	return 0;
}

/*************************************************
��������: OfflineImageExperiment  //

��������: ����ʵ��ģ�飻 //

���������const char *Imgpath ������ʵ��ͼ��·����

���������Infomation *Info �� ����ʵ�����ݣ�//

����ֵ  : bool -- ʵ��ɹ���־λ//

����˵��: ����ʵ��ֻ�ǶԵ���ͼ���������̣�
.		  û��ͼ��ѹ���Ĳ��裻//

*************************************************/
IMGSIMULATION_API bool OfflineImageExperiment(const char *Imgpath, Infomation *Info)
{
	cudaError_t  err;
	int mWidth, mHeight;
	char strFilename[100];
	clock_t start, finish;
	float Difftime;//ʱ���
	float ImageSize;//ͼ��ߴ�
	int ImgChannel;//ͼ��ͨ��

	for (int i = 0; i<5; i++)
	{
		sprintf_s(strFilename, "%s", Imgpath); //��ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ� 
		cudaHostAlloc((void**)&gHostImage[i], gStructVarible.ImgHeight * gStructVarible.ImgWidth * sizeof(unsigned char), cudaHostAllocDefault);
		if (gStructVarible.ImgBitDeep == 24)
		{
			gHostColorImage[i] = new unsigned char[gStructVarible.ImgHeight * gStructVarible.ImgWidth * 3];
		}
		RmwRead8BitBmpFile2Img(strFilename, gHostColorImage[i], gHostImage[i], &mWidth, &mHeight);
	}
	gHostPathImgNumber = 5;//��ʹ�������

						   /****  ��������****/
	CThreadPoolExecutor * pExecutor = new CThreadPoolExecutor();
	pExecutor->Init(1, 1, 1);
	R ExtractPoint;
	RecR RecExtractPoint;

	if (gStructVarible.RecModelFlag == 0)//ȫͼģʽ
	{
		start = clock(); //��ʱ��ʼ
						 /**** �������� ****/
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

		/**** ��ȡ��־����� ****/
		pExecutor->Execute(&ExtractPoint, 0x01);

		pExecutor->Terminate();//��ֹ�߳�
		delete pExecutor;//ɾ���̳߳�	
		finish = clock();//��ʱ����
						 //�õ����μ�¼֮���ʱ���
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
		ImageSize = gStructVarible.ImgHeight * gStructVarible.ImgWidth * ImgChannel / 1024 / 1024;
		Info->ExtractPointSpeed = ImageSize * gHostPathImgNumber * gStructVarible.PictureNum / Difftime;
	}
	else //����ģʽ
	{
		start = clock(); //��ʱ��ʼ
						 /**** �������� ****/
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

		/**** ��ȡ��־����� ****/
		pExecutor->Execute(&RecExtractPoint, 0x01);

		pExecutor->Terminate();
		delete pExecutor;
		finish = clock();//��ʱ����
						 //�õ����μ�¼֮���ʱ���
		Difftime = (float)(finish - start) / CLOCKS_PER_SEC;
		Info->PointNumbers = SignPoint.PointNumbers;
		Info->ExtractPointTimes = Difftime;
		ImgChannel = gStructVarible.ImgBitDeep / 8;//ͼ��ͨ����
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
��������: SinglePictureExtractPoint  //

��������: ����ģʽ��㺯���� //

���������const char *Imgpath  ����ʱ��Ҫ������������ͼ��·��
const char *outputPath ��ȡ���������ļ������·��      //
�����������													//

����ֵ  : ��													//

����˵��: ��������������ļ�����·������Ϊ  outputPath\\OffLine.bin	  //
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

	// �߳����ö���
	dim3 mGrid1(Devpar.ImgMakeborderWidth / 128, Devpar.ImgHeight*Devpar.PictureNum, 1);
	dim3 mGrid2(Devpar.ColThreadNum / 128, Devpar.RowThreadNum, 1);

	//��ȡͼƬ
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
	//�豸���Դ�����
	short *  tDevRecXLeft;
	short *  tDevRecYLeft;
	short *  tDevRecXRight;
	short *  tDevRecYRight;
	short  * tDevLength;
	short  * tDevArea;
	double  *tDevXpos;
	double  *tDevYpos;
	short   *tDevIndex;
	cudaMalloc((void**)&tDevRecXLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//��λ�� xmin
	cudaMalloc((void**)&tDevRecYLeft, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//	    ymin
	cudaMalloc((void**)&tDevRecXRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		xmax
	cudaMalloc((void**)&tDevRecYRight, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//		ymax
	cudaMalloc((void**)&tDevLength, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//�豸�����	�ܳ�
	cudaMalloc((void**)&tDevArea, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//				���
	cudaMalloc((void**)&tDevXpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				xpos
	cudaMalloc((void**)&tDevYpos, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(double));//				ypos
	cudaMalloc((void**)&tDevIndex, Devpar.ColThreadNum*Devpar.RowThreadNum * sizeof(short));//��ȡ������Ч��־
																							//����ռ�����
	short *  tHostRecXLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecYLeft = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecXRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostRecYRight = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostLength = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostArea = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];
	double *  tHostXpos = new double[Devpar.ColThreadNum*Devpar.RowThreadNum];
	double *  tHostYpos = new double[Devpar.ColThreadNum*Devpar.RowThreadNum];
	short *  tHostIndex = new short[Devpar.ColThreadNum*Devpar.RowThreadNum];

	//�˺���ִ��
	if (Devpar.ImgChannelNum == 1)
	{
		cudaMemcpy(tDevGrayImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice);
		//ִ�лҶȻ�����ֵ���˺�������
		GrayMakeBorder << <mGrid1, 128 >> > (tDevGrayImage, tDevpad, Devpar);
	}
	else
	{
		cudaMemcpy(tDevColorImage, tHostImage, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgWidth*Devpar.ImgChannelNum*Devpar.PictureNum, cudaMemcpyHostToDevice);
		ColorMakeBorder << <mGrid1, 128 >> > (tDevColorImage, tDevpad, Devpar);
	}
	//ִ�лҶȻ�����ֵ���˺�������
	Binarization << <mGrid1, 128 >> > (tDevpad, tDev2val, tDevcounter, Devpar);
	//�߽���ȡ
	Dilation << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
	cudaMemcpy(tDev2val, tDevcounter, sizeof(unsigned char)* Devpar.ImgHeight *Devpar.ImgMakeborderWidth*Devpar.PictureNum, cudaMemcpyDeviceToDevice);
	Erosion << <mGrid1, 128 >> > (tDev2val, tDevcounter, Devpar);
	//��ȡ�ܳ��Ͱ�Χ��
	GetCounter << <mGrid2, 128 >> > (tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, Devpar);//��ȡ�����ĺ���
	SelectTrueBox << <mGrid2, 128 >> >(tDevcounter, tDevLength, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevIndex, Devpar);
	SelectNonRepeatBox << <mGrid2, 128 >> > (tDevRecXLeft, tDevRecYLeft, tDevIndex, Devpar);
	GetNonRepeatBox << <mGrid2, 128 >> >(tDevRecXLeft, tDevRecYLeft, tDevIndex, Devpar);
	GetInfo << <mGrid2, 128 >> > (tDevpad, tDevIndex, tDevRecXLeft, tDevRecYLeft, tDevRecXRight, tDevRecYRight, tDevXpos, tDevYpos, tDevArea, Devpar);

	//����������
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
		sprintf(strfilename, "%s\\OffLine.bin", outputPath); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
		fp = fopen(strfilename, "wb");
		fwrite(&myInfo[0], sizeof(CircleInfo)*myInfo.size(), 1, fp);
		fclose(fp);
	}
	//�ͷ��ڴ�
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
��������: DrawPointFlag  //

��������: ��־���ػ棻 //

���������const char *pathBin ����־���������Ϣ�ļ���
.		  const char *pathImg �� ����ͼ��·����//

���������const char *pathWrite �� д��ͼ��·����//

����ֵ  : ��//

����˵��: �����Ѵ�ͼ������ȡ�ı�־���������±��
.		  ��ͼ��ԭλ���ϣ��������Ǻ��ͼ��
.		  �����ʽΪ��ɫʮ����ʽ��//

*************************************************/
IMGSIMULATION_API void DrawPointFlag(const char *pathBin, const char *pathImg, const char *pathWrite)
{
	//��ȡ����
	FILE *fr;
	fr = fopen(pathBin, "rb");

	//��ȡ�ļ���С
	fseek(fr, 0, SEEK_END);//�����ļ�ָ��stream��λ��Ϊ�ļ���β
	long lSize = ftell(fr);//��ȡ���ݳ���
	rewind(fr);//�����ļ�ָ��stream��λ��Ϊ���������ļ���ͷ

			   //��������ռ�
	int FlagSize = lSize / sizeof(CircleInfo);
	CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*FlagSize);

	//��ȡ�ļ�����
	fread(RInfo, sizeof(CircleInfo), FlagSize, fr);
	fclose(fr);

	//���Ʊ�־��ʮ�ּ�
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
��������: Memory_application  //

��������: ȫ���ڴ����룻 //

�����������//

�����������//

����ֵ  : ��//

����˵��: ������ͼ��ߴ����������ȫ���ڴ棻//

*************************************************/
IMGSIMULATION_API void Memory_application()
{
	compress_old_Width = gStructVarible.ImgWidth;
	compress_old_Height = gStructVarible.ImgHeight * gStructVarible.PictureNum;
	//imgWidth = gStructVarible.ImgWidth;						//��ǰ�˽������û�ȡͼƬ����
	//imgHeight = gStructVarible.ImgHeight * gStructVarible.PictureNum;
	compress_imgWidth = (compress_old_Width + 7) / 8 * 8;
	compress_imgHeight = (compress_old_Height + 7) / 8 * 8;
	//�����￪ʼѹ��ע��
	compressratio = gStructVarible.CompressionRatio;		//��ǰ�˽������û�ȡѹ����
	int bmpSize = compress_imgWidth * compress_imgHeight;
	gpHudata = new unsigned char[bmpSize];					//�Ҷ�ͼƬ��ɫ��ֵ��ȷ���ģ���ǰ���ú�
	gpHvdata = new unsigned char[bmpSize];
	memset(gpHudata, 128, compress_imgHeight * compress_imgWidth);
	memset(gpHvdata, 128, compress_imgHeight * compress_imgWidth);
	blocks.x = compress_imgWidth / 8;								//����cudaѹ�������blocksΪ(imgWidth / 8,imgHeight / 8)
	blocks.y = compress_imgHeight / 8;
	blocks.z = 1;
	quantityassgnment();									//��ʼ�������˵�ȫ�ֱ���


	/*�������ݻ�����*/
	//����ɼ�������Ӧ�ڴ�
	gCameraDress= (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength* HardwareParam.DeviceCount * 2);
	for (int i = 0; i < HardwareParam.DeviceCount * 2; i++)
	{
		gCameraBuffer[i] = gCameraDress + i*gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength;

	}
	//ѹ��������
	for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
	{
		gHostComressiongBuffer[i] = (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char) * Bufferlength);
	}

	//������ҳ���ڴ�
	for (int i = 0; i < HardwareParam.DeviceCount + 1; i++)
	{
		cudaHostAlloc((void**)&gHostBuffer[i], gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.ImgChannelNum * sizeof(unsigned char)*Bufferlength, cudaHostAllocDefault);
	}
	//���κ������ڴ�
	gRecupImgData = (unsigned char*)malloc(gStructVarible.ImgWidth*gStructVarible.ImgHeight *gStructVarible.PictureNum*gStructVarible.ImgChannelNum * sizeof(unsigned char));
}

/*************************************************
��������: Memory_release  //

��������: ȫ���ڴ��ͷţ� //

�����������//

�����������//

����ֵ  : ��//

����˵��: ������ͼ��ߴ��ͷ������ȫ���ڴ棻//

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
	delete[]gpHvdata;//qwt����Ҫ���ִ���
}

/*************************************************
��������: SetCameraPar  //

��������: ����������ã� //

���������int ScrBufferlength �� ͼƬ�������ĳ��ȣ�ͼƬ��������
.

���������null��//

����ֵ  : bool -- �������ݳɹ���־λ//

����˵��: �������ڽ��������õĲ�������DLL�Ĳ����У�//

*************************************************/
IMGSIMULATION_API bool SetCameraPar(int ScrBufferlength)
{
	Bufferlength = ScrBufferlength;
	return 0;
}

/*************************************************
��������: SetParameter  //

��������: �������ݣ� //

���������Parameter *info �� �������õĽṹ�������
.		  int len �� �����ݲ���������//

���������Parameter gStructVarible �� ����ʱ�Ľṹ�������//

����ֵ  : bool -- �������ݳɹ���־λ//

����˵��: �������ڽ��������õĲ�������DLL�Ĳ����У�//

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
			ExtractPointSuccess = true;//ʵ�����
		}
		else
		{
			ExtractPointSuccess = false;//��־λ��λ
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
	gStructVarible.ImgChannelNum = gStructVarible.ImgBitDeep / 8;//ͨ����=λ���/8
	//���θ���У��
	if (count == len)
		return true;
	return false;
}

/*************************************************
��������: GetParameter  //

��������: ��ȡ���洫������ṹ�����ֵ�� //

���������Parameter *info �� ���洫������ṹ�壻

���������Parameter gStructVarible �� ����ʱ�Ľṹ�������//

����ֵ  : NULL //

����˵��: �������ڶ�ȡ���洫������ṹ�������//

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
��������: ClearDataCache  //

��������:  ������պ�������DLL���е�ȫ�ֱ�������̬������ȫ�����ó�Ϊ��ʼ��״̬ //

�����������					 //
�����������					//

����ֵ  : ��					//

����˵��: �� //
.

*************************************************/
IMGSIMULATION_API void  ClearDataCache()
{
	//ȫ�ֱ������
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

	//�ྲ̬����
	R::mRindex = 0;
	RecR::mRecindex = 0;
	T::mTindex = 0;
	TC::mTCindex = 0;

	//ȫ��
	Bufferlength = 50;
	ExtractPointInitialSuccessFlag[0] = false;
	ExtractPointInitialSuccessFlag[1] = false;
	ExtractPointInitialSuccessFlag[2] = false;
	ExtractPointSuccess = false;

	//���κ�����
	gHostRecData.clear();
	gRecNum = gHostRecData.size();
	gSingleImgRecNum = gHostRecData.size();
	gRecupImgData = NULL;
	DevUpdateRec[0] = false;
	DevUpdateRec[1] = false;
	DevUpdateRec[2] = false;
	HostUpdateRec = false;
	RecupdataInitialSuccessFlag = false;


	//�������
	for (int i = 0; i < 6; i++)
	{
		BufferBlockIndex[i] = 0;
		gCameraBuffer[i] = false;
		CameraBufferFull[i] = false;
	}

	for (int i = 0; i<4; i++)
	{
		//ҳ���ڴ滺����
		gHostBuffer[i] = NULL;
		PageLockBufferEmpty[i] = true;
		PageLockBufferWorking[i] = false;
		PageLockBufferStartIndex[i] = 0;

		//ѹ������
		gHostComressiongBuffer[i] = NULL;
		gComressionBufferEmpty[i] = true;
		gComressionBufferWorking[i] = false;
		gComressionBufferStartIndex[i] = 0;

	}

}


/****************************��ѹ�����ļ�***************************************************/
/*************************************************
��������: GetFiles  //

��������: ����ĳһ�ļ����ڵ����������ļ���·����
������һ����������ͼ�������������ļ�(.bin�ļ�)�ֽ�Ϊ��������ļ�(.bin),ÿ��ͼƬ��Ӧһ�������ļ� //

���������const char * path �������.bin�ļ����ļ���·��

��������� vector<string>& files �����ļ���֮�ڵ�bin�ļ�·����string��ʽ������	   //

����ֵ  : ��													//

����˵��: �� //
.

*************************************************/
IMGSIMULATION_API void GetFiles(const char * path, vector<string>& files)
{
	//�ļ����  
	intptr_t   hFile = 0;
	//�ļ���Ϣ������һ���洢�ļ���Ϣ�Ľṹ��  
	struct __finddata64_t fileinfo;
	string p;//�ַ��������·��
	if ((hFile = _findfirst64(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//�����ҳɹ��������
	{
		do
		{
			//�����Ŀ¼,����֮�����ļ����ڻ����ļ��У�  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				//�ļ���������"."&&�ļ���������".."
				//.��ʾ��ǰĿ¼
				//..��ʾ��ǰĿ¼�ĸ�Ŀ¼
				//�ж�ʱ�����߶�Ҫ���ԣ���Ȼ�����޵ݹ�������ȥ�ˣ�
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					GetFiles(p.assign(path).append("\\").append(fileinfo.name).c_str(), files);
			}
			//�������,�����б�  
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext64(hFile, &fileinfo) == 0);
		//_findclose������������
		_findclose(hFile);
	}
}

/*************************************************
��������: UnzipFeatureBins  //

��������: ���������������һ����������ͼ�������������ļ�(.bin�ļ�)�ֽ�Ϊ��������ļ�(.bin),ÿ��ͼƬ��Ӧһ�������ļ� //

��������� const char *InputPath  ��������ͼƬ�����������ļ���50�ţ���bufferlenth�йأ�
const char *OutputFilename ��������ļ����ļ���      //
�����������													//

����ֵ  : ��													//

����˵��: �� //
.

*************************************************/
void UnzipFeatureBins(const char *InputPath, const char *OutputFilename)
{
	char strFilename[255];
	FILE *fr;
	fr = fopen(InputPath, "rb");
	if (fr == NULL)//��ͼ��򲻿�����return
	{
		cout << "FILE fail open" << endl;
		return;
	}
	fseek(fr, 0, SEEK_END);
	long lSize = ftell(fr);//��ȡ���ݳ���
	rewind(fr);
	int Datalength = lSize / sizeof(CircleInfo);
	CircleInfo *RInfo = (CircleInfo*)malloc(sizeof(CircleInfo)*Datalength);
	fread(RInfo, sizeof(CircleInfo), Datalength, fr);
	fclose(fr);
	//��ȡ�����ܸ���
	int Dataoffset = 0;
	int Dataindex = 0;
	while (Dataoffset < Datalength)
	{
		CircleInfo mHead = RInfo[Dataoffset];
		Dataoffset++;
		int  mlen = 0;
		if (mHead.area == 0 && int(mHead.xpos) == 99999)//�ж�ͷ����
		{
			Dataindex = mHead.index;
			mlen = mHead.length;
			if (mlen > 0 && Dataoffset + mlen <= Datalength)
			{
				FILE* fp;
				sprintf_s(strFilename, "%s\\%d.bin", OutputFilename, Dataindex); //��3����ͼƬ��·������̬��д�뵽strFilename�����ַ���ڴ�ռ�
				fp = fopen(strFilename, "wb");
				fwrite(&RInfo[Dataoffset], sizeof(CircleInfo)*mlen, 1, fp);
				fclose(fp);
				Dataoffset = Dataoffset + mlen;
			}
		}
	}
};

/*************************************************
��������: UnzipFeatureFiles  //

��������: �����������ĳһ���ļ����µĴ������ļ�����������ͼƬ�������ļ�.bin�ļ�����ѹ�ɵ��������ļ���
��GPU����������ɵ������ļ���һ�������ļ�����bufferlenth��ͼ���������ú���������������ļ�
�ֽ�ɶ��bufferlenth��ͼƬ��

��������� const char *Filepath  ���������ļ��ĵ��ļ���

�����������													//

����ֵ  : ��													//

����˵��: �� //
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

//��ѹͼƬ����
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



