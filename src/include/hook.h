#ifndef __HOOK_H__
#define __HOOK_H__

typedef struct {
  void *fn_ptr;   // function pointer
  char *name;     //function name
} entry_t;

#endif // __HOOK_H__