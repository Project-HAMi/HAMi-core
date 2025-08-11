#ifndef __HOOK_H__
#define __HOOK_H__

typedef struct {
  void *fn_ptr;   // function pointer
  char *name;     //function name
} entry_t;

#define OVERRIDE_ENUM(x) OVERRIDE_##x

#define FIND_ENTRY(table, sym) ({ (table)[OVERRIDE_ENUM(sym)].fn_ptr; })

#endif // __HOOK_H__