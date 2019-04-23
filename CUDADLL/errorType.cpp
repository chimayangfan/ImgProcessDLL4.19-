#include "errorType.h"
#include <stdio.h>  

G_CVAR ErrorType ERR_OK			= { 0, "Success" };//�ɹ�
G_CVAR ErrorType ERR_DISKSPACE	= { 1, "Insufficent hard disk space." };//Ӳ�̿ռ䲻��
G_CVAR ErrorType ERR_CUDADEVICE	= { 2, "Insufficent availabe GPU device" };//�޿���GPU�豸
G_CVAR ErrorType ERR_CONFIG		= { 3, "Incorrect parameter configuration" };//������������

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
		/**����Ϳ���ֱ��ʹ�ô������͵� ��desc�� ��Ա��������ô�����Ϣ���ַ������൱���� */
		printf("Exception :%d, %s\n", ret.code, ret.desc);
	}

	return ret;
}
