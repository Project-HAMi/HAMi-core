/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 The HAMi Authors.
 */

#ifndef SRC_INCLUDE_HOSTPID_BROKER_H_
#define SRC_INCLUDE_HOSTPID_BROKER_H_

#include <sys/types.h>

#define HOSTPID_BROKER_SOCKET_PATH "/tmp/vgpulock/hostpid/broker.sock"

int hostpid_broker_enabled(const char *value);
int hostpid_broker_query(const char *socket_path, pid_t *host_pid);
int hostpid_broker_query_trusted(const char *socket_path, pid_t *host_pid);
int hostpid_broker_validate_trust(const char *socket_path,
                                  uid_t trusted_uid,
                                  int require_readonly);

#endif  // SRC_INCLUDE_HOSTPID_BROKER_H_
