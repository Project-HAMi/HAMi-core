#!/usr/bin/env bash
# Runs the host PID discovery race test against libvgpu.so.
# Phase 1: probes start one after another while a process without libvgpu
#          keeps appearing on and leaving the device.
# Phase 2: several probes start at once, each with its own cache file, as
#          separate containers do.
# Each probe must find its own PID. Exit 77 = no GPU, or not in the host PID
# namespace (CTest SKIP_RETURN_CODE).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${LIBVGPU:-$ROOT/build/libvgpu.so}"
BIN="${TEST_BIN:-$ROOT/build/test/test_hostpid_race}"
LIMIT="${CUDA_DEVICE_MEMORY_LIMIT:-2048m}"
PROBES="${HAMI_HOSTPID_PROBES:-20}"
CONCURRENT="${HAMI_HOSTPID_CONCURRENT:-8}"
ROUNDS="${HAMI_HOSTPID_ROUNDS:-3}"
NEIGHBOUR_MIB="${HAMI_HOSTPID_NEIGHBOUR_MIB:-512}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "SKIP: no NVIDIA GPU available (nvidia-smi)" >&2
  exit 77
fi
# Probe's getpid() is compared with the host PID libvgpu found.
if [[ "$(awk '/^NSpid:/ {print NF - 1}' /proc/self/status)" != "1" ]]; then
  echo "SKIP: needs the host PID namespace (docker run --pid=host)" >&2
  exit 77
fi
if [[ ! -f "$LIB" ]]; then
  echo "missing $LIB — build first: (cd \"$ROOT\" && ./build.sh)" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — rebuild so test/CMakeLists.txt picks up the new test" >&2
  exit 1
fi

WORK="$(mktemp -d)"
NEIGHBOUR_PID=""
cleanup() {
  [[ -n "$NEIGHBOUR_PID" ]] && kill "$NEIGHBOUR_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p /tmp/vgpulock

probe() {  # probe <log> <cache>
  env LD_PRELOAD="$LIB" CUDA_DEVICE_MEMORY_LIMIT="$LIMIT" \
      CUDA_DEVICE_MEMORY_SHARED_CACHE="$2" LIBCUDA_LOG_LEVEL=3 \
      "$BIN" probe >"$1" 2>&1 || true
}

WRONG=0
judge() {  # judge <phase> <log>...
  local phase="$1" ok=0 wrong=0 missing=0 log pid host
  shift
  for log in "$@"; do
    pid="$(sed -n 's/^probe pid=\([0-9]*\).*/\1/p' "$log")"
    host="$(sed -n 's/.*hostPid=\([0-9]*\).*/\1/p' "$log" | tail -n 1)"
    if [[ -z "$pid" || -z "$host" ]]; then
      missing=$((missing + 1))
    elif [[ "$pid" == "$host" ]]; then
      ok=$((ok + 1))
    else
      wrong=$((wrong + 1))
      echo "  $phase: probe $pid was given host PID $host:"
      grep -E 'probe pid=|Primary Context Size|OOM' "$log" | sed 's/^/    /'
    fi
  done
  echo "$phase: $ok found their own PID, $wrong took another process's, $missing without a host PID"
  WRONG=$((WRONG + wrong))
}

echo "Running $BIN with LD_PRELOAD=$LIB, LIMIT=$LIMIT"

env -u LD_PRELOAD "$BIN" neighbour 600 "$NEIGHBOUR_MIB" >"$WORK/neighbour.log" 2>&1 &
NEIGHBOUR_PID=$!
sleep 2
for i in $(seq 1 "$PROBES"); do
  probe "$WORK/seq.$i.log" "$WORK/seq.cache"
done
kill "$NEIGHBOUR_PID" 2>/dev/null || true
wait "$NEIGHBOUR_PID" 2>/dev/null || true
NEIGHBOUR_PID=""
judge "phase 1 (neighbour without libvgpu, $PROBES probes)" "$WORK"/seq.*.log

for r in $(seq 1 "$ROUNDS"); do
  for i in $(seq 1 "$CONCURRENT"); do
    probe "$WORK/conc.$r.$i.log" "$WORK/conc.$r.$i.cache" &
  done
  wait
done
judge "phase 2 ($ROUNDS x $CONCURRENT probes at once, separate caches)" "$WORK"/conc.*.log

if [[ "$WRONG" -ne 0 ]]; then
  echo "FAIL: $WRONG probes took another process's host PID"
  exit 1
fi
echo "PASS"
