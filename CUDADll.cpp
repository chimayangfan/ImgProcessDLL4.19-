// CUDADll.cpp: 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "cudakernel.h"

int _stdcall CUDAadd(int a,int b)
{
	return (a + b);
}