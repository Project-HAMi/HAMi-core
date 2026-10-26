#ifndef __NVML_PREFIX__
#define __NVML_PREFIX__

#define NVML_NO_UNVERSIONED_FUNC_DEFS

typedef struct {
  void *fn_ptr;
  char *name;
} entry_t;

extern entry_t nvml_library_entry[];

#endif