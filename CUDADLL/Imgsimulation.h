#pragma once
//声明相关宏
#define IMGSIMULATION_EXPORTS 1
#ifdef IMGSIMULATION_EXPORTS
#define IMGSIMULATION_API __declspec(dllexport)
#else
#define IMGSIMULATION_API __declspec(dllimport)
#endif
#define EXTERN_C extern "C"


//参数设置
EXTERN_C IMGSIMULATION_API void Image_path_check(const char *path, const char *exten);
EXTERN_C IMGSIMULATION_API void SimulationImageTest(const char *path, int mWidth, int mHeight);
EXTERN_C IMGSIMULATION_API void Memory_application();
EXTERN_C IMGSIMULATION_API void Memory_release();
