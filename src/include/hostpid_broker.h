#ifndef SRC_INCLUDE_HOSTPID_BROKER_H_
#define SRC_INCLUDE_HOSTPID_BROKER_H_

#include <sys/types.h>

#define HOSTPID_BROKER_SOCKET_PATH "/tmp/vgpulock/hostpid/broker.sock"

int hostpid_broker_enabled(const char *value);
/* Raw protocol exchange for tests or callers providing their own trust policy.
 * Production initialization must use hostpid_broker_query_trusted instead.
 * Both queries return 0 on success, or -1 with errno and *host_pid set to 0.
 * One 500 ms deadline covers socket I/O. Deferred cancellation closes the
 * socket; asynchronous cancellation is unsupported. Neither query runs NVML
 * discovery or updates CUDA accounting.
 */
int hostpid_broker_query(const char *socket_path, pid_t *host_pid);
/* Requires the fixed socket path, root ownership, a read-only immediate
 * directory and a connected server with UID 0. No request is sent unless
 * those checks pass. Peer credentials are currently supported only on Linux.
 */
int hostpid_broker_query_trusted(const char *socket_path, pid_t *host_pid);
int hostpid_broker_validate_trust(const char *socket_path,
                                  uid_t trusted_uid,
                                  int require_readonly);

#ifdef HOSTPID_BROKER_TESTING
/* Tests observe real EAGAIN before allowing the server to drain its queue. */
void hostpid_broker_test_connect_retry(int fd);
/* Test the production peer check on a temporary socket without root mounts. */
int hostpid_broker_test_query_peer(const char *socket_path, pid_t *host_pid,
                                  uid_t trusted_uid);
#endif

#endif  // SRC_INCLUDE_HOSTPID_BROKER_H_
