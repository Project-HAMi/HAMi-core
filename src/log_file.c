// Minimal file to provide fp1 for logging
// This is used by standalone tools like shrreg-tool that don't need the full utils.c
#include "include/log_utils.h"

// Define fp1 for logging (declared extern in log_utils.h)
// This is defined here so all executables (libsoftmig.so, shrreg-tool, etc.) can use it
FILE *fp1 = NULL;

