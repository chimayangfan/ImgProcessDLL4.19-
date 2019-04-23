#pragma once
#ifndef __ERRORTYPE_HPP__  
#define __ERRORTYPE_HPP__ 
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
#define DLL_EXPORT
#endif

#if defined (DLL_EXPORT)
#define G_DLL_EXPORT __declspec(dllexport)
#else
#define G_DLL_EXPORT __declspec(dllimport)

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

G_CVAR ErrorType ERR_OK;//成功
G_CVAR ErrorType ERR_DISKSPACE;//硬盘空间不足
G_CVAR ErrorType ERR_CUDADEVICE;//无可用GPU设备
G_CVAR ErrorType ERR_CONFIG;//参数配置有误

#endif