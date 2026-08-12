#!/usr/bin/env bash
# Run the cross-process oom_check TOCTOU reproducer against libvgpu.so.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="${LIBVGPU:-$ROOT/build/libvgpu.so}"
BIN="${TEST_BIN:-$ROOT/build/test/test_concurrent_oom_race}"
CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/hami_oom_race.cache}"
LIMIT="${CUDA_DEVICE_MEMORY_LIMIT:-1024m}"
ALLOC="${HAMI_RACE_ALLOC:-600m}"
ROUNDS="${HAMI_RACE_ROUNDS:-30}"
WINDOW_US="${HAMI_ALLOC_RACE_WINDOW_US:-20000}"

if [[ ! -f "$LIB" ]]; then
  echo "missing $LIB — build first: (cd \"$ROOT\" && ./build.sh)" >&2
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — rebuild so test/CMakeLists.txt picks up the new test" >&2
  exit 1
fi

mkdir -p /tmp/vgpulock
rm -f "$CACHE"

export LD_PRELOAD="$LIB"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="$CACHE"
export CUDA_DEVICE_MEMORY_LIMIT="$LIMIT"
export HAMI_RACE_ALLOC="$ALLOC"
export HAMI_RACE_ROUNDS="$ROUNDS"
export HAMI_ALLOC_RACE_WINDOW_US="$WINDOW_US"
export LIBCUDA_LOG_LEVEL="${LIBCUDA_LOG_LEVEL:-1}"

echo "Running: $BIN $*"
echo "  LD_PRELOAD=$LD_PRELOAD"
echo "  LIMIT=$LIMIT ALLOC=$ALLOC ROUNDS=$ROUNDS WINDOW_US=$WINDOW_US"
echo "  CACHE=$CACHE"
exec "$BIN" "$@"
