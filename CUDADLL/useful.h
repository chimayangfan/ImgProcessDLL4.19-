/******************************************************************
Copyright:川大电子信息学院黄勇韬
Date:2018-10-10
Description:定义压缩程序所需要的常量、结构体和一部分功能函数
*****************************************************************/


#ifndef __USEFUL_HEADER__
#define __USEFUL__HEADER__

#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <iostream>

/*-----------------主机端定义的标准亮度量化表------------------*/
const unsigned char gHostLuminance_Quantization_Table[64] =
{
	16,  11,  10,  16,  24,  40,  51,  61,
	12,  12,  14,  19,  26,  58,  60,  55,
	14,  13,  16,  24,  40,  57,  69,  56,
	14,  17,  22,  29,  51,  87,  80,  62,
	18,  22,  37,  56,  68, 109, 103,  77,
	24,  35,  55,  64,  81, 104, 113,  92,
	49,  64,  78,  87, 103, 121, 120, 101,
	72,  92,  95,  98, 112, 100, 103,  99
};

/*-----------------主机端定义的标准色差量化表------------------*/
const unsigned char gHostChrominance_Quantization_Table[64] =
{
	17,  18,  24,  47,  99,  99,  99,  99,
	18,  21,  26,  66,  99,  99,  99,  99,
	24,  26,  56,  99,  99,  99,  99,  99,
	47,  66,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99
};

/*-------------对8*8的GPU block块中的值的取值顺序-------------*/
const char gHostZigZag[64] =
{
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63
};

/*-----------------------标准直流亮度霍夫曼编码表-------------*/
const char gHostStandard_DC_Luminance_NRCodes[] = { 0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };		//1个bit的编码0个，2个bit的编码0个，3个bit的编码7个
const unsigned char gHostStandard_DC_Luminance_Values[] = { 4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11 };			//需要编码的十进制数

/*-----------------------标准直流色差霍夫曼编码表-------------*/																											/*-----------------------标准直流色差霍夫曼编码表-------------*/
const char gHostStandard_DC_Chrominance_NRCodes[] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };		//1个bit的编码0个，2个bit的编码3个，3个bit的编码1个
const unsigned char gHostStandard_DC_Chrominance_Values[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };		//需要编码的十进制数

/*-----------------------标准交流亮度霍夫曼编码表-------------*/																											/*-----------------------标准交流亮度霍夫曼编码表-------------*/
const char gHostStandard_AC_Luminance_NRCodes[] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };	//1个bit的编码0个，2个bit的编码2个，3个bit的编码1个
const unsigned char gHostStandard_AC_Luminance_Values[] =													//需要编码的十六进制数
{
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
	0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
	0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
	0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
	0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
	0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
	0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
	0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};

/*-----------------------标准交流色差霍夫曼编码表-------------*/
const char gHostStandard_AC_Chrominance_NRCodes[] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };	//1个bit的编码0个，2个bit的编码2个，3个bit的编码1个
const unsigned char gHostStandard_AC_Chrominance_Values[] =													//需要编码的十六进制数
{
	0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
	0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
	0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
	0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
	0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
	0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
	0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
	0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
	0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
	0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};


/*-----------------------磁盘上.jpg图片的结构头-------------*/
struct FrameHeader
{
	unsigned char nSamplePrecision;
	unsigned short nHeight;
	unsigned short nWidth;
	unsigned char nComponents;
	unsigned char aComponentIdentifier[3];
	unsigned char aSamplingFactors[3];
	unsigned char aQuantizationTableSelector[3];
};

/*-----------------------磁盘上.jpg图片的扫描头-------------*/
struct ScanHeader
{
	unsigned char nComponents;
	unsigned char aComponentSelector[3];
	unsigned char aHuffmanTablesSelector[3];
	unsigned char nSs;
	unsigned char nSe;
	unsigned char nA;
};

/*------------------磁盘上.jpg图片中储存的量化表-------------*/
struct QuantizationTable
{
	unsigned char nPrecisionAndIdentifier;
	unsigned char aTable[64];
};

/*-------------磁盘上.jpg图片中储存的霍夫曼编码表-------------*/
struct HuffmanTable
{
	unsigned char nClassAndIdentifier;
	unsigned char aCodes[16];
	unsigned char aTable[256];
};



int DivUp(int x, int d)									//计算x/d的值并返回
{
	return (x + d - 1) / d;
}

template<typename T>
T readAndAdvance(const unsigned char *&pData)			//读pData中的一个值
{
	T nElement = readBigEndian<T>(pData);
	pData += sizeof(T);
	return nElement;
}

template<typename T>
void writeAndAdvance(unsigned char *&pData, T nElement)	//写pData处的一个值为nElement
{
	writeBigEndian<T>(pData, nElement);
	pData += sizeof(T);
}


int nextMarker(const unsigned char *pData, int &nPos, int nLength)
{
	unsigned char c = pData[nPos++];

	do
	{
		while (c != 0xffu && nPos < nLength)
		{
			c = pData[nPos++];
		}

		if (nPos >= nLength)
			return -1;

		c = pData[nPos++];
	} while (c == 0 || c == 0x0ffu);

	return c;
}

void writeMarker(unsigned char nMarker, unsigned char *&pData)	//写pData处的一个值为nMarker
{
	*pData++ = 0x0ff;
	*pData++ = nMarker;
}

void writeWords(unsigned short nMarker, unsigned char *&pData)	//写pData处的一个值为nMarker
{
	*pData++ = (nMarker >> 8);
	*pData++ = nMarker;
}

void writeChar(unsigned char nMarker, unsigned char *&pData)	//写pData处的一个值为nMarker
{
	*pData++ = nMarker;
}

void writeJFIFTag(unsigned char *&pData)						//在pData中写一个JFIF标志
{
	const char JFIF_TAG[] =
	{
		0x4a, 0x46, 0x49, 0x46, 0x00,
		0x01, 0x02,
		0x00,
		0x00, 0x01, 0x00, 0x01,
		0x00, 0x00
	};

	writeMarker(0x0e0, pData);
	writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
	memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
	pData += sizeof(JFIF_TAG);
}


void readFrameHeader(const unsigned char *pData, FrameHeader &header)		//从pData中读出.jpg结构头，并保存到header处
{
	readAndAdvance<unsigned short>(pData);
	header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
	header.nHeight = readAndAdvance<unsigned short>(pData);
	header.nWidth = readAndAdvance<unsigned short>(pData);
	header.nComponents = readAndAdvance<unsigned char>(pData);

	for (int c = 0; c<header.nComponents; ++c)
	{
		header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
		header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
		header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
	}

}

void writeFrameHeader(const FrameHeader &header, unsigned char *&pData)		//把header处的.jpg结构头写到pData中
{
	unsigned char aTemp[128];
	unsigned char *pTemp = aTemp;

	writeAndAdvance<unsigned char>(pTemp, header.nSamplePrecision);
	writeAndAdvance<unsigned short>(pTemp, header.nHeight);
	writeAndAdvance<unsigned short>(pTemp, header.nWidth);
	writeAndAdvance<unsigned char>(pTemp, header.nComponents);

	for (int c = 0; c<header.nComponents; ++c)
	{
		writeAndAdvance<unsigned char>(pTemp, header.aComponentIdentifier[c]);
		writeAndAdvance<unsigned char>(pTemp, header.aSamplingFactors[c]);
		writeAndAdvance<unsigned char>(pTemp, header.aQuantizationTableSelector[c]);
	}

	unsigned short nLength = (unsigned short)(pTemp - aTemp);

	writeMarker(0x0C0, pData);
	writeAndAdvance<unsigned short>(pData, nLength + 2);
	memcpy(pData, aTemp, nLength);
	pData += nLength;
}


void readScanHeader(const unsigned char *pData, ScanHeader &header)		//从pData中读出.jpg扫描头，并保存到header处
{
	readAndAdvance<unsigned short>(pData);

	header.nComponents = readAndAdvance<unsigned char>(pData);

	for (int c = 0; c<header.nComponents; ++c)
	{
		header.aComponentSelector[c] = readAndAdvance<unsigned char>(pData);
		header.aHuffmanTablesSelector[c] = readAndAdvance<unsigned char>(pData);
	}

	header.nSs = readAndAdvance<unsigned char>(pData);
	header.nSe = readAndAdvance<unsigned char>(pData);
	header.nA = readAndAdvance<unsigned char>(pData);
}


void writeScanHeader(const ScanHeader &header, unsigned char *&pData)		//把header处的.jpg扫描头写到pData中
{
	unsigned char aTemp[128];
	unsigned char *pTemp = aTemp;

	writeAndAdvance<unsigned char>(pTemp, header.nComponents);

	for (int c = 0; c<header.nComponents; ++c)
	{
		writeAndAdvance<unsigned char>(pTemp, header.aComponentSelector[c]);
		writeAndAdvance<unsigned char>(pTemp, header.aHuffmanTablesSelector[c]);
	}

	writeAndAdvance<unsigned char>(pTemp, header.nSs);
	writeAndAdvance<unsigned char>(pTemp, header.nSe);
	writeAndAdvance<unsigned char>(pTemp, header.nA);

	unsigned short nLength = (unsigned short)(pTemp - aTemp);

	writeMarker(0x0DA, pData);
	writeAndAdvance<unsigned short>(pData, nLength + 2);
	memcpy(pData, aTemp, nLength);
	pData += nLength;
}


void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables)	//从pData中读出量化表，并保存到pTables处
{
	unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

	while (nLength > 0)
	{
		unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>(pData);
		int nIdentifier = nPrecisionAndIdentifier & 0x0f;

		pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
		memcpy(pTables[nIdentifier].aTable, pData, 64);
		pData += 64;

		nLength -= 65;
	}
}

void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData)	//把table处的量化表写到pData中
{
	writeMarker(0x0DB, pData);
	writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
	memcpy(pData, &table, sizeof(QuantizationTable));
	pData += sizeof(QuantizationTable);
}

void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables)			//从pData中读出霍夫曼编码表，并保存到pTables处
{
	unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

	while (nLength > 0)
	{
		unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>(pData);
		int nClass = nClassAndIdentifier >> 4; // AC or DC
		int nIdentifier = nClassAndIdentifier & 0x0f;
		int nIdx = nClass * 2 + nIdentifier;
		pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;
		printf("%d -------- *** %d\n", nIdx, nClassAndIdentifier);

		// Number of Codes for Bit Lengths [1..16]
		int nCodeCount = 0;

		for (int i = 0; i < 16; ++i)
		{
			pTables[nIdx].aCodes[i] = readAndAdvance<unsigned char>(pData);
			nCodeCount += pTables[nIdx].aCodes[i];
		}

		memcpy(pTables[nIdx].aTable, pData, nCodeCount);
		pData += nCodeCount;

		nLength -= (17 + nCodeCount);
	}
}

void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData)			//把table处的霍夫曼编码表写到pData中
{
	writeMarker(0x0C4, pData);

	// Number of Codes for Bit Lengths [1..16]
	int nCodeCount = 0;

	for (int i = 0; i < 16; ++i)
	{
		nCodeCount += table.aCodes[i];
	}

	writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
	memcpy(pData, &table, 17 + nCodeCount);
	pData += 17 + nCodeCount;
}


int compressratio;											//设置压缩比的变量
static unsigned char gHostYTable[64];						//经过处理的亮度量化表，之后会传送到GPU中
static unsigned char gHostCbCrTable[64];					//经过处理的色差量化表，之后会传送到GPU中
int compress_imgWidth;										//压缩图片宽度
int compress_imgHeight;										//压缩图片高度
int compress_old_Width;
int compress_old_Height;
//unsigned char* udata;										记得修改一下，删除后调试看是否报错
//unsigned char* vdata;										记得修改一下，删除后调试看是否报错
HuffmanTable aHuffmanTables[4];								//主机端的霍夫曼编码表
HuffmanTable *pHuffmanDCTables = aHuffmanTables;
HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
QuantizationTable aQuantizationTables[4];					//主机端的量化表
FrameHeader oFrameHeader;									//.jpg图片结构头，一次实验产生的所有图片共用
ScanHeader oScanHeader;										//.jpg图片扫描头，一次实验产生的所有图片共用
size_t dataduiqi[3];										//GPU显存有特殊的对齐要求，用来存放对齐后的数据宽度
//unsigned char* totalinfo = new unsigned char[1078];		记得修改一下，删除后调试看是否报错

/**************************************************************************************************
Function:		quantityassgnment()
Description:   初始化主机端的全局变量，包括gHostYTable、gHostCbCrTable、oFrameHeader、oScanHeader
aHuffmanTables、aQuantizationTables

Calls:          无
Input:          无

Output:         无
Return:         无
**************************************************************************************************/
void quantityassgnment()
{
	for (int i = 0; i<64; i++)								//根据设置的压缩比初始化gHostYTable[64]、gHostCbCrTable[64]
	{
		int temp = ((int)(gHostLuminance_Quantization_Table[i] * 50 + compressratio) / 100);
		if (temp <= 0) temp = 1;
		if (temp>0xFF) temp = 0xFF;
		gHostYTable[gHostZigZag[i]] = (unsigned char)temp;

		temp = ((int)(gHostChrominance_Quantization_Table[i] * 50 + compressratio) / 100);
		if (temp <= 0) 	temp = 1;
		if (temp>0xFF) temp = 0xFF;
		gHostCbCrTable[gHostZigZag[i]] = (unsigned char)temp;
	}
	memset(&oFrameHeader, 0, sizeof(FrameHeader));
	memset(aQuantizationTables, 0, 4 * sizeof(QuantizationTable));
	memset(aHuffmanTables, 0, 4 * sizeof(HuffmanTable));

	oFrameHeader.nSamplePrecision = 8;						//初始化.jpg图片结构头
	oFrameHeader.nHeight = compress_imgHeight;
	oFrameHeader.nWidth = compress_imgWidth;
	oFrameHeader.nComponents = 3;
	oFrameHeader.aComponentIdentifier[0] = 1;
	oFrameHeader.aComponentIdentifier[1] = 2;
	oFrameHeader.aComponentIdentifier[2] = 3;
	oFrameHeader.aSamplingFactors[0] = 17;
	oFrameHeader.aSamplingFactors[1] = 17;
	oFrameHeader.aSamplingFactors[2] = 17;
	oFrameHeader.aQuantizationTableSelector[0] = 0;
	oFrameHeader.aQuantizationTableSelector[1] = 1;
	oFrameHeader.aQuantizationTableSelector[2] = 1;

	aQuantizationTables[0].nPrecisionAndIdentifier = 0;		//初始化主机端的量化表
	memcpy(aQuantizationTables[0].aTable, gHostYTable, 64);
	aQuantizationTables[1].nPrecisionAndIdentifier = 1;
	memcpy(aQuantizationTables[1].aTable, gHostCbCrTable, 64);


	aHuffmanTables[0].nClassAndIdentifier = 0;				//初始化主机端的霍夫曼编码表
	memcpy(aHuffmanTables[0].aCodes, gHostStandard_DC_Luminance_NRCodes, 16);
	memcpy(aHuffmanTables[0].aTable, gHostStandard_DC_Luminance_Values, 12);
	aHuffmanTables[2].nClassAndIdentifier = 16;
	memcpy(aHuffmanTables[2].aCodes, gHostStandard_AC_Luminance_NRCodes, 16);
	memcpy(aHuffmanTables[2].aTable, gHostStandard_AC_Luminance_Values, 162);

	aHuffmanTables[1].nClassAndIdentifier = 1;
	memcpy(aHuffmanTables[1].aCodes, gHostStandard_DC_Chrominance_NRCodes, 16);
	memcpy(aHuffmanTables[1].aTable, gHostStandard_DC_Chrominance_Values, 12);
	aHuffmanTables[3].nClassAndIdentifier = 17;
	memcpy(aHuffmanTables[3].aCodes, gHostStandard_AC_Chrominance_NRCodes, 16);
	memcpy(aHuffmanTables[3].aTable, gHostStandard_AC_Chrominance_Values, 162);

	oScanHeader.nComponents = 3;							//初始化.jpg图片扫描头
	oScanHeader.aComponentSelector[0] = 1;
	oScanHeader.aComponentSelector[1] = 2;
	oScanHeader.aComponentSelector[2] = 3;
	oScanHeader.aHuffmanTablesSelector[0] = 0;
	oScanHeader.aHuffmanTablesSelector[1] = 17;
	oScanHeader.aHuffmanTablesSelector[2] = 17;
	oScanHeader.nSs = 0;
	oScanHeader.nA = 0;
	oScanHeader.nSe = 63;
}
#endif