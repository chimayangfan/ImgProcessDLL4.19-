#include "errorType.h"
#include <stdio.h>  

G_CVAR ErrorType ERR_OK			= { 0, "Success" };//成功
G_CVAR ErrorType ERR_DISKSPACE	= { 1, "Insufficent hard disk space." };//硬盘空间不足
G_CVAR ErrorType ERR_CUDADEVICE	= { 2, "Insufficent availabe GPU device" };//无可用GPU设备
G_CVAR ErrorType ERR_CONFIG		= { 3, "Incorrect parameter configuration" };//参数配置有误

ErrorType Exception(int code)
{
	ErrorType ret = ERR_OK;

	switch (code) {
		case 0: ret = ERR_OK; break;
		case 1: ret = ERR_DISKSPACE; break;
		case 2: ret = ERR_CUDADEVICE; break;
		case 3: ret = ERR_CONFIG; break;
	}

	if (ret != ERR_OK)
	{
		/**这里就可以直接使用错误类型的 ‘desc’ 成员变量来获得错误信息的字符串，相当方便 */
		printf("Exception :%d, %s\n", ret.code, ret.desc);
	}

	return ret;
}
