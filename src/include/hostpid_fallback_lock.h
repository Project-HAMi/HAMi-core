#ifndef SRC_INCLUDE_HOSTPID_FALLBACK_LOCK_H_
#define SRC_INCLUDE_HOSTPID_FALLBACK_LOCK_H_

#include <sys/types.h>
#include <time.h>

#define HOSTPID_FALLBACK_LOCK_PATH "/tmp/vgpulock/hostpid"
#ifndef HOSTPID_FALLBACK_LOCK_TIMEOUT_MS
#define HOSTPID_FALLBACK_LOCK_TIMEOUT_MS 30000U
#endif

int hostpid_fallback_lock_acquire(void);
int hostpid_fallback_lock_deadline_after_ms(struct timespec *deadline,
                                            unsigned int timeout_ms);
int hostpid_fallback_lock_acquire_until(const struct timespec *deadline);
int hostpid_fallback_lock_acquire_at(const char *path, uid_t trusted_owner,
                                     unsigned int timeout_ms);
int hostpid_fallback_lock_acquire_at_until(
    const char *path, uid_t trusted_owner, const struct timespec *deadline);
int hostpid_fallback_lock_release(void);
void hostpid_fallback_lock_after_fork(void);
int hostpid_fallback_lock_active_fd(void);

#endif  // SRC_INCLUDE_HOSTPID_FALLBACK_LOCK_H_
