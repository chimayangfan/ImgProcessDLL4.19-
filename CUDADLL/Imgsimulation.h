#pragma once
//������غ�
#define IMGSIMULATION_EXPORTS 1
#ifdef IMGSIMULATION_EXPORTS
#define IMGSIMULATION_API __declspec(dllexport)
#else
#define IMGSIMULATION_API __declspec(dllimport)
#endif
#define EXTERN_C extern "C"

//���洫������ṹ��
IMGSIMULATION_API typedef struct
{
	//const char *ImgReadPath; /*>1<*///ͼ���ȡ·��
	//const char *ImgSavePath; /*>2<*///ͼ�񱣴�·��
	//const char *DataReadPath;/*>3<*///���ݱ���·��
	char ImgReadPath[100];	 /*>1<*///ͼ���ȡ·��
	char ImgSavePath[100];	 /*>2<*///ͼ�񱣴�·��
	char DataReadPath[100];	 /*>3<*///���ݱ���·��
	int ImgBitDeep;          /*>4<*///ͼ��λ��
	int ImgChannelNum;       /*>5<*///ͼ��ͨ����
	int ImgHeight;           /*>6<*///����
	int ImgWidth;            /*>7<*///����
	int ImgMakeborderWidth;  /*>8<*///������غ�Ŀ��=-=
	int Threshold;           /*>9<*///��ֵ������ֵ 
	int LengthMin;           /*>10<*///�ܳ�����Сֵ 
	int LengthMax;           /*>11<*///�ܳ������ֵ
	int PicBlockSize;        /*>12<*///GPU�߳̿�ߴ�
	int ColThreadNum;        /*>13<*///�з���ֿ���������������ǿ�����ģ� ���㹫ʽΪ Devpar.ColThreadNum=(ImgWidth/8+127)/128*128; =-=
	int RowThreadNum;        /*>14<*///�з��������(�з���Ŀ�������������128 ��������)=-=
	int AreaMin;             /*>15<*///�����ֵ��Сֵ
	int AreaMax;             /*>16<*///�����ֵ���ֵ
	int CompressionRatio;    /*>17<*///ͼ��ѹ����
	int PictureNum;          /*>18<*///ƴͼ����   =-=
	int TerminateFlag;       /*>19<*///�Ƿ���ֹʵ��
	int RecModelFlag;        /*>20<*///����ģʽ��־λ false
	int RecPadding;          /*>21<*///��Χ�����������Ŀ
}Parameter;

//���ؽ�����Ϣ�ṹ��
IMGSIMULATION_API typedef struct
{
	int DeviceCount;		 /*>1<*/// GPU�豸��
	int ColThreadNum;		 /*>2<*///�з���ֿ���������������ǿ�����ģ�
	int RowThreadNum;		 /*>3<*///�з��������(�з���Ŀ�������������128 ��������)
	int ImgHeight;			 /*>4<*///����
	int ImgWidth;			 /*>5<*///����
	int ImgMakeborderWidth;	 /*>6<*///������غ�Ŀ��
	int Threshold;			 /*>7<*///��ֵ������ֵ 
	int LengthMin;			 /*>8<*///�ܳ�����Сֵ 
	int LengthMax;			 /*>9<*///�ܳ������ֵ
	int ThreadNum;			 /*>10<*///GPU�����߳���
	int CPUThreadCount;		 /*>11<*///CPU�����߳���
	int ImgProcessingNumbers;/*>12<*///����ͼ������
	int PointNumbers;		 /*>13<*///��־������
	float ExtractPointTimes; /*>14<*///��ȡ��־����ʱ
	float ExtractPointSpeed; /*>15<*///��ȡ��־���ٶ�
	float CompressionTimes;	 /*>16<*///ѹ����ʱ
	float CompressionSpeed;	 /*>17<*///ѹ���ٶ�
	float SynchronizeTimes;	 /*>18<*///ͬ�������ʱ
	float SynchronizeSpeed;	 /*>19<*///ͬ�������ٶ�
}Infomation;

//Ӳ���豸�ṹ��
IMGSIMULATION_API typedef struct
{
	int DeviceCount;		/*>1<*/// GPU�豸��
	int GpuId;				/*>2<*///�豸��  �����Ӳ���ϵ�Gpu��Id��
	int DeviceID;			/*>3<*///�豸���  �����ʾ�������豸��ţ���һ��������GPUΪ0���ڶ���Ϊ1��������Ϊ2��
	int regsPerBlock;		/*>4<*///�߳̿����ʹ�õ�32λ�Ĵ��������ֵ���ദ�����ϵ������߳̿����ͬʱʵ����Щ�Ĵ���
	int maxThreadsPerBlock;	/*>5<*///ÿ����������߳���
	int major;				/*>6<*///����������������
	int minor;				/*>7<*///���������Ĵ�Ҫ����
	int deviceOverlap;		/*>8<*///�����Ƿ���ͬʱִ��cudaMemcpy()�������ĺ��Ĵ���
	int multiProcessorCount;/*>9<*///�豸�϶ദ����������
	int ExPointThreads;		/*>10<*///CPU��ȡ��־���߳���
	int CompThreads;		/*>11<*///CPUͼ��ѹ���߳���
	int CUDAStreamNum;		/*>12<*///CUDA����Ŀ
	int DiskRemainingSpace;	/*>13<*///����ʣ������
}HardwareInfo;

//���������Ľӿ�
EXTERN_C IMGSIMULATION_API int HardwareInit(HardwareInfo *HardwareProp);
EXTERN_C IMGSIMULATION_API int Image_Pretreatment(const char *path, const char *exten, int ChooseMode);
EXTERN_C IMGSIMULATION_API bool SimulationImageTest(const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API void  SimulationTestReport(const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API bool SimulationExperient(int ChooseMode);
EXTERN_C IMGSIMULATION_API void  SimulationTestSynchronize(const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API void  SimulationTestExtractPoint(const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API void  SimulationTestComression(const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API bool OnlineImageExperiment(int ChooseMode, const char *path, Infomation *Info);
EXTERN_C IMGSIMULATION_API bool OnlineImageRecExperiment(int ChooseMode, Infomation *Info);
EXTERN_C IMGSIMULATION_API int OnlineImageRefresh(unsigned char *pImg);
EXTERN_C IMGSIMULATION_API bool OfflineImageExperiment(const char *Imgpath, Infomation *Info);
EXTERN_C IMGSIMULATION_API bool SinglePictureExtractPoint(const char *Imgpath, const char*outputPath);
EXTERN_C IMGSIMULATION_API void DrawPointFlag(const char *pathBin, const char *pathImg, const char *pathWrite);
EXTERN_C IMGSIMULATION_API void Memory_application();
EXTERN_C IMGSIMULATION_API void Memory_release();
EXTERN_C IMGSIMULATION_API bool SetCameraPar(int Bufferlength);
EXTERN_C IMGSIMULATION_API bool SetParameter(Parameter *info, int len);
EXTERN_C IMGSIMULATION_API void GetParameter(Parameter *info);
EXTERN_C IMGSIMULATION_API void UnzipPictureFiles(const char * Filepath);


