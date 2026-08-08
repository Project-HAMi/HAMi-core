/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#ifndef HOSTPID_FALLBACK_LOCK_H
#define HOSTPID_FALLBACK_LOCK_H

#include <sys/types.h>

#define HOSTPID_FALLBACK_LOCK_PATH "/tmp/vgpulock"

int hostpid_fallback_lock_acquire(void);
int hostpid_fallback_lock_acquire_at(const char *path, uid_t trusted_owner,
                                     unsigned int timeout_ms);
int hostpid_fallback_lock_release(void);
void hostpid_fallback_lock_after_fork(void);
int hostpid_fallback_lock_active_fd(void);

#endif
