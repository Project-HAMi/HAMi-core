#!/bin/bash
# Run the full HAMi-core initialization benchmark suite.
# Produces the measurements shown in issue #1662.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the core binaries are built. phase_probe is optional (only built
# when NVML is available), so ask make which targets it actually produces
# instead of hardcoding it as required.
BUILT_TARGETS="$(make print-targets)"
HAVE_PHASE_PROBE=0
case " $BUILT_TARGETS " in
    *" phase_probe "*) HAVE_PHASE_PROBE=1 ;;
esac

if [ ! -f bench_init ] || [ ! -f nested_retain ] || [ ! -f warm_holder ] || [ ! -f abi_check ] || \
   { [ "$HAVE_PHASE_PROBE" = 1 ] && [ ! -f phase_probe ]; }; then
    echo "Building benchmarks..."
    make clean
    make
fi

LIBVGPU_SO="${SCRIPT_DIR}/../build/libvgpu.so"
if [ ! -r "$LIBVGPU_SO" ]; then
    echo "error: $LIBVGPU_SO not found or not readable" >&2
    echo "       build it first (see the parent directory's build instructions);" >&2
    echo "       an invalid LD_PRELOAD would be silently ignored by the loader," >&2
    echo "       producing native-CUDA results mislabeled as HAMi-core measurements." >&2
    exit 1
fi
export LD_PRELOAD="$LIBVGPU_SO"
export CUDA_DEVICE_MEMORY_LIMIT=2048m
export CUDA_DEVICE_SM_LIMIT=100
export LIBCUDA_LOG_LEVEL=3

HOLDER_PID=""
cleanup() {
    if [ -n "$HOLDER_PID" ]; then
        kill "$HOLDER_PID" 2>/dev/null || true
        wait "$HOLDER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Clean up any stale cache
rm -f /tmp/cudevshr.cache

echo "==============================================="
echo "HAMi-core Initialization Benchmark Suite"
echo "==============================================="
echo

echo "1. Measuring phase breakdown (set_task_pid bottleneck identification)"
if [ "$HAVE_PHASE_PROBE" = 1 ]; then
    echo "   Running: phase_probe"
    ./phase_probe
else
    echo "   Skipping: phase_probe (libnvidia-ml not found, NVML unavailable)"
fi
echo

echo "2. Testing cost of holding a primary context"
echo "   Running: nested_retain"
./nested_retain
echo

echo "3. Measuring concurrent initialization scaling"
echo "   (keeping a primary context holder alive)"
echo "   Starting warm_holder..."
./warm_holder primary >/dev/null 2>&1 &
HOLDER_PID=$!
sleep 1

echo "   Running bench_init at N=1,2,4,8,16,32,64,128..."
for n in 1 2 4 8 16 32 64 128; do
    echo ""
    echo "   N=$n processes:"
    ./bench_init "$n" 2>&1 | grep -E "p50|p95|p99|max|Total"
done

cleanup
HOLDER_PID=""
echo
echo "   warm_holder stopped."
echo

echo "4. Verifying ABI compatibility"
echo "   Running: abi_check"
./abi_check
echo

echo "==============================================="
echo "Benchmark suite complete."
echo "See README.md for expected results and context."
echo "==============================================="
