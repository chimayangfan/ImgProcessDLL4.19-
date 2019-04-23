#pragma once
#ifndef __ERRORTYPE_HPP__  
#define __ERRORTYPE_HPP__ 

#include <stdio.h>  

/************************************************************************/
/** 这部分是为了兼容visual studio和gcc对于动态库类型导出修饰符 */
/** macro define for special tool chain. */
#if defined(__GNUC__) /** !!!for gcc */
////////////////////////////////////////////////////////////////////////////
#define G_GCC_VERSION(maj,min) \
        ((__GNUC__ > (maj)) || (__GNUC__ == (maj) && __GNUC_MINOR__ >= (min)))

/** define API export  macro */
#if G_GCC_VERSION(4,0) //for gcc(version >= 4.0)
#define G_DLL_EXPORT __attribute__((visibility("default")))
#else
#define G_DLL_EXPORT
#endif

#elif defined(_MSC_VER) /** !!!for visual studio */
////////////////////////////////////////////////////////////////////////////

/** auto define API export macro */
#if !defined(DLL_EXPORT) && defined(_WINDLL) //todo,process _USRDLL,_AFXDLL
//#define DLL_EXPORT//此处动态库与正常项目相反
#endif

#if defined (DLL_EXPORT)
#define G_DLL_EXPORT __declspec(dllimport)
#else
#define G_DLL_EXPORT __declspec(dllexport)

#endif

#else /** !!!for unknown tool chain */
///////////////////////////////////////////////////////////////////////////
#error "!!unspport this toolchain at now!!"
#endif /** !!! end of tool chain define */


/** general global variable decorate macro(for external reference ) */
#define G_VAR  extern G_DLL_EXPORT 

/** general const global variable decorate macro(for external reference ) */
#define G_CVAR G_VAR const 
/************************************************************************/

/** error type */
struct ErrorType
{
	int code = 0;  /** error code( or number) */
	const char* desc = ""; /** the describe of error, aways not null */

	ErrorType(int cd, const char* dsc)
		:code(cd), desc(dsc)
	{
	}

	inline bool operator==(const ErrorType& et) const
	{
		return (code == et.code);
	}

	inline bool operator!=(const ErrorType& et) const
	{
		return (code != et.code);
	}

	inline ErrorType& operator=(const ErrorType& et)
	{
		code = et.code;
		desc = et.desc;

		return *this;
	}

};

G_CVAR ErrorType ERR_OK				= { 0, "Success" };//成功
G_CVAR ErrorType ERR_DISKSPACE		= { 1, "Insufficent hard disk space." };//硬盘空间不足
G_CVAR ErrorType ERR_CUDADEVICE		= { 2, "Insufficent availabe GPU device" };//无可用GPU设备
G_CVAR ErrorType ERR_CONFIG			= { 3, "Incorrect parameter configuration" };//参数配置有误

ErrorType CustomizeException(int code)
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
		printf("Exception %d: %s\n", ret.code, ret.desc);
	}

	return ret;
}

#endif