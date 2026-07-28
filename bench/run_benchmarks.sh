#!/bin/bash
# Run the full HAMi-core initialization benchmark suite.
# Produces the measurements shown in issue #1662.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure binaries are built
if [ ! -f bench_init ] || [ ! -f phase_probe ] || [ ! -f warm_holder ]; then
    echo "Building benchmarks..."
    make clean
    make
fi

export LD_PRELOAD="${SCRIPT_DIR}/../build/libvgpu.so"
export CUDA_DEVICE_MEMORY_LIMIT=2048m
export CUDA_DEVICE_SM_LIMIT=100
export LIBCUDA_LOG_LEVEL=3

# Clean up any stale cache
rm -f /tmp/cudevshr.cache

echo "==============================================="
echo "HAMi-core Initialization Benchmark Suite"
echo "==============================================="
echo

echo "1. Measuring phase breakdown (set_task_pid bottleneck identification)"
echo "   Running: phase_probe"
./phase_probe
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
    ./bench_init "$n" 2>&1 | grep -E "p50|p90|p99|max|Total"
done

kill $HOLDER_PID 2>/dev/null || true
wait $HOLDER_PID 2>/dev/null || true
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
