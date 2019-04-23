#pragma once
//声明相关宏
#define IMGSIMULATION_EXPORTS 1
#ifdef IMGSIMULATION_EXPORTS
#define IMGSIMULATION_API __declspec(dllexport)
#else
#define IMGSIMULATION_API __declspec(dllimport)
#endif
#define EXTERN_C extern "C"

//界面传入参数结构体
IMGSIMULATION_API typedef struct
{
	//const char *ImgReadPath; /*>1<*///图像读取路径
	//const char *ImgSavePath; /*>2<*///图像保存路径
	//const char *DataReadPath;/*>3<*///数据保存路径
	char ImgReadPath[100];	 /*>1<*///图像读取路径
	char ImgSavePath[100];	 /*>2<*///图像保存路径
	char DataReadPath[100];	 /*>3<*///数据保存路径
	int ImgBitDeep;          /*>4<*///图像位深
	int ImgChannelNum;       /*>5<*///图像通道数
	int ImgHeight;           /*>6<*///行数
	int ImgWidth;            /*>7<*///列数
	int ImgMakeborderWidth;  /*>8<*///填充像素后的宽度=-=
	int Threshold;           /*>9<*///二值化的阈值 
	int LengthMin;           /*>10<*///周长的最小值 
	int LengthMax;           /*>11<*///周长的最大值
	int PicBlockSize;        /*>12<*///GPU线程块尺寸
	int ColThreadNum;        /*>13<*///列方向分块数量（这个数量是块填充后的） 计算公式为 Devpar.ColThreadNum=(ImgWidth/8+127)/128*128; =-=
	int RowThreadNum;        /*>14<*///行方向块数量(行方向的块数量不用填充成128 的整数倍)=-=
	int AreaMin;             /*>15<*///面积阈值最小值
	int AreaMax;             /*>16<*///面积阈值最大值
	int CompressionRatio;    /*>17<*///图像压缩比
	int PictureNum;          /*>18<*///拼图数量   =-=
	int TerminateFlag;       /*>19<*///是否终止实验
	int RecModelFlag;        /*>20<*///矩形模式标志位 false
	int RecPadding;          /*>21<*///包围盒填充像素数目
}Parameter;

//返回界面信息结构体
IMGSIMULATION_API typedef struct
{
	int DeviceCount;		 /*>1<*/// GPU设备数
	int ColThreadNum;		 /*>2<*///列方向分块数量（这个数量是块填充后的）
	int RowThreadNum;		 /*>3<*///行方向块数量(行方向的块数量不用填充成128 的整数倍)
	int ImgHeight;			 /*>4<*///列数
	int ImgWidth;			 /*>5<*///行数
	int ImgMakeborderWidth;	 /*>6<*///填充像素后的宽度
	int Threshold;			 /*>7<*///二值化的阈值 
	int LengthMin;			 /*>8<*///周长的最小值 
	int LengthMax;			 /*>9<*///周长的最大值
	int ThreadNum;			 /*>10<*///GPU配置线程数
	int CPUThreadCount;		 /*>11<*///CPU配置线程数
	int ImgProcessingNumbers;/*>12<*///处理图像数量
	int PointNumbers;		 /*>13<*///标志点数量
	float ExtractPointTimes; /*>14<*///提取标志点用时
	float ExtractPointSpeed; /*>15<*///提取标志点速度
	float CompressionTimes;	 /*>16<*///压缩用时
	float CompressionSpeed;	 /*>17<*///压缩速度
	float SynchronizeTimes;	 /*>18<*///同步处理耗时
	float SynchronizeSpeed;	 /*>19<*///同步处理速度
}Infomation;

//硬件设备结构体
IMGSIMULATION_API typedef struct
{
	int DeviceCount;		/*>1<*/// GPU设备数
	int GpuId;				/*>2<*///设备号  这个是硬件上的Gpu的Id号
	int DeviceID;			/*>3<*///设备编号  这个表示启动的设备编号（第一个启动的GPU为0，第二个为1；第三个为2）
	int regsPerBlock;		/*>4<*///线程块可以使用的32位寄存器的最大值，多处理器上的所有线程快可以同时实用这些寄存器
	int maxThreadsPerBlock;	/*>5<*///每个块中最大线程数
	int major;				/*>6<*///计算能力的主代号
	int minor;				/*>7<*///计算能力的次要代号
	int deviceOverlap;		/*>8<*///器件是否能同时执行cudaMemcpy()和器件的核心代码
	int multiProcessorCount;/*>9<*///设备上多处理器的数量
	int ExPointThreads;		/*>10<*///CPU提取标志点线程数
	int CompThreads;		/*>11<*///CPU图像压缩线程数
	int CUDAStreamNum;		/*>12<*///CUDA流数目
	int DiskRemainingSpace;	/*>13<*///磁盘剩余容量
}HardwareInfo;

//向外声明的接口
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


