#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "huffman.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include<iostream>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
#define SHIFT              3  
#define MASK               0x7 
#define CODE_THREADS       128
#define Prefix_SumThreads  128
#define DCT_BLOCK_WIDTH    64
#define DCT_BLOCK_HEIGHT   8
#define DCT_BLOCK_SIZE     8
#define THREAD_WARP        32
#define HALF_THREAD_WARP   16
#define DCT_SHEMEM_STRIDE  65
#define DCT_SHEMEM_HEIGHT  8
#define PRESUM_THREADS     256
	typedef short int BSI16;
	typedef int            BOOL;
	typedef unsigned char  BYTE;
	typedef unsigned short WORD;
	typedef unsigned long  DWORD;
	/*----------------------------DCT�任ʹ�õ��Ĳ���---------------------------*/
#define C_norm   (float)0.35355f // 1 / (8^0.5)
#define C_a      (float)1.38703f // a = (2^0.5) * cos(    pi / 16);����DCT���任  
#define C_b      (float)1.30656f // b = (2^0.5) * cos(    pi /  8);����DCT���任  
#define C_c      (float)1.17587f // c = (2^0.5) * cos(3 * pi / 16);����DCT���任 
#define C_d      (float)0.78569f // d = (2^0.5) * cos(5 * pi / 16);����DCT���任   
#define C_e      (float)0.541196f // e = (2^0.5) * cos(3 * pi /  8);����DCT���任 
#define C_f      (float)0.275899f // f = (2^0.5) * cos(7 * pi / 16);����DCT���任  

	/*-----------------------------RGBתYUV����---------------------------------*/
#define C_Yr    (float)0.299f
#define C_Yg    (float)0.587f
#define C_Yb    (float)0.114f
#define C_Ur    (float)-0.169f
#define C_Ug    (float)-0.331f
#define C_Ub    (float)0.5f
#define C_Vr    (float)0.5f
#define C_Vg    (float)-0.419f
#define C_Vb    (float)-0.081f

	/*-----------------------------RGBתYUV�����Ҷ�ͼר��---------------------------------*/
	__constant__ const BSI16  d_pColorTable[256] = { 0 };

	/*************************************zigzagɨ���*************************************/
	int aZIGZAG[64] =
	{
		0, 1, 5, 6, 14, 15, 27, 28,
		2, 4, 7, 13, 16, 26, 29, 42,
		3, 8, 12, 17, 25, 30, 41, 43,
		9, 11, 18, 24, 31, 40, 44, 53,
		10, 19, 23, 32, 39, 45, 52, 54,
		20, 22, 33, 38, 46, 51, 55, 60,
		21, 34, 37, 47, 50, 56, 59, 61,
		35, 36, 48, 49, 57, 58, 62, 63,
	};
	const int ZIGZAG[64] =                         //����jpegͼ�������ɨ���
	{
		0, 1, 8, 16, 9, 2, 3, 10,
		17, 24, 32, 25, 18, 11, 4, 5,
		12, 19, 26, 33, 40, 48, 41, 34,
		27, 20, 13, 6, 7, 14, 21, 28,
		35, 42, 49, 56, 57, 50, 43, 36,
		29, 22, 15, 23, 30, 37, 44, 51,
		58, 59, 52, 45, 38, 31, 39, 46,
		53, 60, 61, 54, 47, 55, 62, 63,
	};

	/***************************************��׼������***********************************/
	int STD_QUANT_TAB_LUMIN[64] =
	{
		16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99,
	};

	int STD_QUANT_TAB_CHROM[64] =
	{
		16, 18, 24, 47, 99, 99, 99, 99,
		18, 21, 26, 66, 99, 99, 99, 99,
		24, 26, 56, 99, 99, 99, 99, 99,
		47, 66, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
	};
	/* ���Ͷ��� */
	/* ����������Ͷ��� */
	/*--------------------------------ѹ������GPU��Ҫ�Ľṹ��needmemory-------------------------------*/
	typedef struct gpuneedmemory
	{
		BSI16  *d_ydst;			  //GPU��y����DCT�任�󻺴���Դ�ռ�
		BYTE   *d_JPEGdata;       //�Ҷ�ͼGPU�ڱ�������ʹ�õĸ����ռ����GPU�ڱ�������Y����ʹ�õĸ����ռ�
		BYTE   *last_JPEGdata;    //���յı�������bit��
		BYTE   *d_bsrc;           //�Դ��е�ԭʼλͼ���ݴ�ſռ䣬ÿ���Ǳ����ֽڶ��룬ʹ���Դ��ά�ռ���亯��
		int    *prefix_num;       //���ڻ����һ��ǰ׺�������
		int    *last_prefix_num;       //���ڻ���ڶ���ǰ׺�������
		int    *d_blocksum;       //ǰ׺��͸����ռ�
		int    *dc_component;      //ǰ׺��͸����ռ�
		int    *d_datalen;        //GPU�����ڼ�¼�������ֽڳ��ȣ����䵽CPU cpuneedmemory.dst_JPEGdatalength��								//pDScan�ĳ�ʼ����С
	};
	/*-------------------------------CPU��ʹ�õ���ͼ�������ڴ�----------------------------------*/
	typedef struct cpuneedmemory {
		BYTE   *h_obsrc;             //CPU�ϵ�ԭʼͼ�����ݴ洢�ռ�
		BYTE   *pDstJpeg;            //CPU�ϵ�JPGͼ�����ݻ���ռ�
		BYTE   *pDstJpegDataStart;   //CPU�ϵ�JPGͼ�����ݻ���ռ�,ͼ�������׵�ַ
		BYTE   *pDstOutput;          //ͼ�������ƶ���ַָ��
		int    dst_JPEGdatalength;   //���յı���������BYTE����
	};
	/*--------------------------------ѹ������GPU��Ҫ�ĳ�������-------------------------------*/
	typedef struct needconstdata
	{
		float *DEV_STD_QUANT_TAB_CHROM;  //����UV������������ʱ�õ��ĳ���
		float *DEV_STD_QUANT_TAB_LUMIN;  //����Y������������ʱ�õ��ĳ���
		int   *DEV_ZIGZAG;               //zigzagɨ���
		Npp32s nScanSize;
	};
	/**************��¼ͼ��ԭʼ�߶ȺͿ��**********************************************************/
	//** StrideF = ��width + 3��& ~3
	typedef struct
	{
		int   width;             /* ��� */
		int   height;            /* �߶� */
		size_t   StrideF;
	} RIM;
#ifdef __cplusplus
}
#endif

#endif
