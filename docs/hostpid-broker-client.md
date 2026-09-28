# Host PID broker client

This module implements the version 1 broker protocol from
[HAMi's device plugin](https://github.com/Project-HAMi/HAMi/pull/2417).
Nothing calls it yet, so it is not linked into `libvgpu`; only its test builds it.
This split does not change host PID discovery, NVML fallback or CUDA accounting.

## API and trust

`hostpid_broker_query_trusted()` accepts only
`/tmp/vgpulock/hostpid/broker.sock`. Before sending a request, it requires:

1. A real immediate directory owned by root, without group or other write access.
2. A real Unix socket owned by root.
3. A read only mount for that directory.
4. UID 0 on the connected peer, checked with Linux `SO_PEERCRED`.

An unsupported peer credential API fails with `ENOTSUP`. The directory and
socket checks use `lstat`; they reject symlinks at those two locations. They do
not validate every ancestor. The mount and parent directory must be supplied by
the trusted device plugin. The connected peer check is separate from pathname
metadata checks.

`hostpid_broker_query()` performs a raw exchange at a supplied path. It does not
authenticate the server and is intended for tests or callers with their own
trust policy. Production initialization must use the trusted query.

Both functions return 0 and a positive host PID on success. On failure they
return -1, set `errno` and clear the output PID. A null output pointer is rejected
with `EINVAL`. Neither function chooses or runs a fallback.

`hostpid_broker_enabled()` recognizes only the exact string `1`. It does not read
the environment or enable any behavior by itself.

## Protocol and deadline

The request is eight bytes: `HPID`, a big endian 16 bit version of 1 and a
16 bit command of 1. The response is twelve bytes: `HPID`, version 1, a 16 bit
status and a 32 bit PID. Only status 0 and a PID from 1 through `INT_MAX` are
accepted. Invalid magic, version or status returns `EPROTO`; an out of range PID
returns `ERANGE`. A partial response followed by EOF returns `ECONNRESET`.

One 500 millisecond monotonic deadline covers connection, request writing and
response reading. Receiving a partial response does not renew it. On Linux, a
full accept queue returns `EAGAIN`; the client retries `connect()` with waits
starting at 1 millisecond and capped at 10 milliseconds under the same deadline.
Filesystem trust checks happen before this socket deadline and have no separate
timeout.
Socket errors are returned to the caller. Descriptors are nonblocking and have
`FD_CLOEXEC`; ordinary returns and deferred thread cancellation close them.
Asynchronous cancellation is unsupported.

## Tests

`test/test_hostpid_broker.c` needs no CUDA toolkit, NVIDIA driver or GPU. It is
part of the normal test build, so `make build-in-docker` runs it with the rest of
CTest. To run it alone after a build:

```sh
ctest --test-dir build --output-on-failure -R '^hostpid_broker$'
```

The test build uses a 100 millisecond socket deadline. The cases cover valid and
fragmented replies, malformed protocol fields, invalid PIDs, early EOF, silent
and trickling servers, invalid inputs, path trust failures, peer credentials and
cancellation cleanup. Linux cases fill the accept queue and check recovery after
`EAGAIN`, timeout and cancellation during retries. Peer acceptance and UID
mismatch cases run on Linux. Other systems report that coverage as skipped and
check that unsupported peer validation fails closed.

These tests do not exercise the fixed production mount, Kubernetes allocation,
a GPU workload or init integration.
