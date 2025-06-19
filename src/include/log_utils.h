#ifndef __LOG_UTILS_H__
#define __LOG_UTILS_H__

#include <libgen.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "fp1_global.h"


#ifdef FILEDEBUG 
#define LOG_DEBUG(msg, ...) { \
    if ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=4)) {\
        if (fp1==NULL) fp1 = fopen ("/tmp/vgpulog", "a"); \
        fprintf(fp1, "[HAMI-core Debug(%d:%ld:%s:%d)]: "msg"\n",getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
        }\
    }
#define LOG_INFO(msg, ...) { \
    if ( \
         /*(getenv("LIBCUDA_LOG_LEVEL")==NULL) || */\
         (getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=3)) {\
        if (fp1==NULL) fp1 = fopen ("/tmp/vgpulog", "a"); \
        fprintf(fp1, "[HAMI-core Info(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
         }\
    }
#define LOG_WARN(msg, ...) { \
    if ( \
        (getenv("LIBCUDA_LOG_LEVEL")==NULL) || \
        ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=2))) {\
        if (fp1==NULL) fp1 = fopen ("/tmp/vgpulog", "a"); \
        fprintf(fp1, "[HAMI-core Warn(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
        }\
    }
#define LOG_MSG(msg, ...) { \
    if ( \
        (getenv("LIBCUDA_LOG_LEVEL")==NULL) || \
        ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=2))) {\
        if (fp1==NULL) fp1 = fopen ("/tmp/vgpulog", "a"); \
        fprintf(fp1, "[HAMI-core Msg(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
         }\
    }
#define LOG_ERROR(msg, ...) { \
    if (fp1==NULL) fp1 = fopen ("/tmp/vgpulog", "a"); \
    fprintf(fp1, "[HAMI-core ERROR (pid:%d thread=%ld %s:%d)]: "msg"\n", getpid(), pthread_self(), basename(__FILE__),__LINE__, ##__VA_ARGS__); \
}
#else
#define LOG_DEBUG(msg, ...) { \
    if ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=4)) {\
        fprintf(stderr, "[HAMI-core Debug(%d:%ld:%s:%d)]: "msg"\n",getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
         }\
    }
#define LOG_INFO(msg, ...) { \
    if ( \
        (getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=3)) {\
        fprintf(stderr, "[HAMI-core Info(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
        }\
    }
#define LOG_WARN(msg, ...) { \
    if ( \
        (getenv("LIBCUDA_LOG_LEVEL")==NULL) || \
        ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=2))) {\
        fprintf(stderr, "[HAMI-core Warn(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
        }\
    }
#define LOG_MSG(msg, ...) { \
    if ( \
        (getenv("LIBCUDA_LOG_LEVEL")==NULL) || \
        ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=2))) {\
        fprintf(stderr, "[HAMI-core Msg(%d:%ld:%s:%d)]: "msg"\n", getpid(),pthread_self(),basename(__FILE__),__LINE__,##__VA_ARGS__); \
         }\
    }
#define LOG_ERROR(msg, ...) { \
    fprintf(stderr, "[HAMI-core ERROR (pid:%d thread=%ld %s:%d)]: "msg"\n", getpid(), pthread_self(), basename(__FILE__),__LINE__, ##__VA_ARGS__); \
}
#endif

#define CHECK_DRV_API(f)  {                   \
    CUresult status = (f);                    \
    if (status != CUDA_SUCCESS) {             \
        LOG_WARN("Driver error at %d: %d",   \
            __LINE__, status);                \
        return status;                        \
    } }                                       \

#define CHECK_NVML_API(f)  {                  \
    nvmlReturn_t status = (f);                \
    if (status != NVML_SUCCESS) {             \
        LOG_WARN("NVML error at line %d: %d",    \
            __LINE__, status);                \
        return status;                        \
    } }                                       \

#define CHECK_CU_RESULT(res)  {               \
    if (res != CUDA_SUCCESS) {                \
        LOG_WARN("Driver error at %d: %d",   \
            __LINE__, res);                   \
        return res;                           \
    } }                                       \

#define CHECK_SUCCESS(res) {                  \
    if (res != CUDA_SUCCESS)                  \
        return res;                           \
}

#define IF_CHECK_OOM(res) {                   \
    if (res < 0)                              \
        return CUDA_ERROR_OUT_OF_MEMORY;      \
}     


#endif