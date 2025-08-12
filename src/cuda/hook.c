#include "include/libcuda_hook.h"
#include <string.h>
#include "include/libvgpu.h"
#include "include/multi_func_hook.h"
#include "include/hook.h"

typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;

entry_t cuda_library_entry[]= {
    #define X(func) {.name = #func, .fn_ptr = NULL},
    CUDA_FUNCTIONS(X)
    X(cuMemGetInfo_v2)
    #undef X

};

_Static_assert(sizeof(cuda_library_entry)/sizeof(entry_t) == CUDA_ENTRY_END,
             "cuda_library_entry size doesn't match cuda_override_enum_t");

int prior_function(char tmp[500]) {
    char *pos = tmp + strlen(tmp) - 3;
    if (pos[0]=='_' && pos[1]=='v') {
        if (pos[2]=='2')
            pos[0]='\0';
        else
            pos[2]--;
        return 1;
    }
    return 0;
}

void load_cuda_libraries() {
    void *table = NULL;
    int i = 0;
    char cuda_filename[FILENAME_MAX];
    char tmpfunc[500];

    LOG_INFO("Start hijacking");

    snprintf(cuda_filename, FILENAME_MAX - 1, "%s","libcuda.so.1");
    cuda_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(cuda_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", cuda_filename);
    }

    for (i = 0; i < CUDA_ENTRY_END; i++) {
        LOG_DEBUG("LOADING %s %d",cuda_library_entry[i].name,i);
        cuda_library_entry[i].fn_ptr = real_dlsym(table, cuda_library_entry[i].name);
        if (!cuda_library_entry[i].fn_ptr) {
            cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,cuda_library_entry[i].name);
            if (!cuda_library_entry[i].fn_ptr){
                LOG_INFO("can't find function %s in %s", cuda_library_entry[i].name,cuda_filename);
                memset(tmpfunc,0,500);
                strcpy(tmpfunc,cuda_library_entry[i].name);
                while (prior_function(tmpfunc)) {
                    cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,tmpfunc);
                    if (cuda_library_entry[i].fn_ptr) {
                        LOG_INFO("found prior function %s",tmpfunc);
                        break;
                    } 
                }
            }
        }
    }
    LOG_INFO("loaded_cuda_libraries");
    if (cuda_library_entry[0].fn_ptr==NULL){
        LOG_WARN("is NULL");
    }
    dlclose(table);
}


// find func by cuda version
const char* get_real_func_name(const char* base_name,int cuda_version) {
  int i = 0;
  for (i = 0; i < sizeof(g_func_map)/sizeof(g_func_map[0]); ++i) {
    CudaFuncMapEntry *entry = &g_func_map[i];
    // check fun name
    if (strcmp(entry->func_name, base_name) != 0) continue;
    // check cuda version
    if (cuda_version >= entry->min_ver && cuda_version <= entry->max_ver) {
      return entry->real_name;
    }
  }
  return NULL; // if not found
}

void* find_real_symbols_in_table(const char *symbol) {
  void *pfn;
  //this symbol always has suffix like _v2,_v3
  pfn = __dlsym_hook_section(NULL,symbol);
  if (pfn!=NULL) {
    return pfn;
  }
  return NULL;
}

void *find_symbols_in_table(const char *symbol) {
    char symbol_v[500];
    void *pfn;
    strcpy(symbol_v,symbol);
    strcat(symbol_v,"_v3");
    pfn = __dlsym_hook_section(NULL,symbol_v);
    if (pfn!=NULL) {
        return pfn;
    }
    symbol_v[strlen(symbol_v)-1]='2';
    pfn = __dlsym_hook_section(NULL,symbol_v);
    if (pfn!=NULL) {
        return pfn;
    }
    pfn = __dlsym_hook_section(NULL,symbol);
    if (pfn!=NULL) {
        return pfn;
    }
    return NULL;
}

void *find_symbols_in_table_by_cudaversion(const char *symbol,int  cudaVersion) {
  void *pfn;
  const char *real_symbol;
  real_symbol = get_real_func_name(symbol,cudaVersion);
  if (real_symbol == NULL) {
    // if not find in mulit func version def, use origin logic
    pfn = find_symbols_in_table(symbol);
  } else {
    pfn = find_real_symbols_in_table(real_symbol);
  }
  return pfn;
}


CUresult (*cuGetProcAddress_real) ( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags ); 

CUresult _cuGetProcAddress ( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags ) {
    LOG_INFO("into _cuGetProcAddress symbol=%s:%d",symbol,cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn==NULL){
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress,symbol,pfn,cudaVersion,flags);
        return res;
    }else{
        LOG_DEBUG("found symbol %s",symbol);
        return CUDA_SUCCESS;
    }
}

CUresult cuGetProcAddress ( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags ) {
    LOG_INFO("into cuGetProcAddress symbol=%s:%d",symbol,cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol,"cuGetProcAddress")==0) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress,symbol,pfn,cudaVersion,flags); 
        if (res==CUDA_SUCCESS) {
            cuGetProcAddress_real=*pfn;
            *pfn=_cuGetProcAddress;
        }
        return res;
    }
    if (*pfn==NULL){
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress,symbol,pfn,cudaVersion,flags);
        return res;
    }else{
        LOG_DEBUG("found symbol %s",symbol);
        return CUDA_SUCCESS;
    }
}

CUresult _cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus){
    LOG_INFO("into _cuGetProcAddress_v2 symbol=%s:%d",symbol,cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn==NULL){
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress_v2,symbol,pfn,cudaVersion,flags,symbolStatus);
        return res;
    }else{
        LOG_DEBUG("found symbol %s",symbol);
        return CUDA_SUCCESS;
    } 
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus){
    LOG_INFO("into cuGetProcAddress_v2 symbol=%s:%d",symbol,cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol,"cuGetProcAddress_v2")==0) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress_v2,symbol,pfn,cudaVersion,flags,symbolStatus); 
        if (res==CUDA_SUCCESS) {
            cuGetProcAddress_real=*pfn;
            *pfn=_cuGetProcAddress_v2;
        }
        return res;
    }
    if (*pfn==NULL){
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress_v2,symbol,pfn,cudaVersion,flags,symbolStatus);
        return res;
    }else{
        LOG_DEBUG("found symbol %s",symbol);
        void *optr;
        return CUDA_OVERRIDE_CALL(cuda_library_entry,cuGetProcAddress_v2,symbol,&optr,cudaVersion,flags,symbolStatus);
    } 
}