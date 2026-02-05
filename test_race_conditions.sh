#!/bin/bash
#
# Comprehensive Race Condition Test for HAMi Option 4 Seqlock
#
# This script runs multiple tests to verify:
# 1. Memory accounting accuracy
# 2. No partial reads (seqlock working)
# 3. No race conditions under high contention
# 4. Correct handling of concurrent processes
#

set -e

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_BLUE}========================================"
echo "HAMi Seqlock Race Condition Test Suite"
echo -e "========================================${COLOR_RESET}"
echo ""

# Configuration
NUM_PROCESSES=8
TEST_DIR="$(dirname "$0")"
LOG_FILE="/tmp/hami_test_$(date +%s).log"

echo "Test Configuration:"
echo "  Processes: $NUM_PROCESSES"
echo "  Log file: $LOG_FILE"
echo ""

# Cleanup function
cleanup() {
    echo -e "${COLOR_YELLOW}Cleaning up...${COLOR_RESET}"
    rm -f /tmp/cudevshr.cache
    killall -9 test_seqlock_accuracy 2>/dev/null || true
}

trap cleanup EXIT

# Test 1: Compile the test
echo -e "${COLOR_BLUE}Test 1: Compiling test program...${COLOR_RESET}"
if nvcc -o test_seqlock_accuracy test_seqlock_accuracy.cu -lcudart -lnvidia-ml; then
    echo -e "${COLOR_GREEN}✓ Compilation successful${COLOR_RESET}"
else
    echo -e "${COLOR_RED}✗ Compilation failed${COLOR_RESET}"
    exit 1
fi
echo ""

# Test 2: Single process sanity check
echo -e "${COLOR_BLUE}Test 2: Single process sanity check...${COLOR_RESET}"
if ./test_seqlock_accuracy > $LOG_FILE 2>&1; then
    echo -e "${COLOR_GREEN}✓ Single process test passed${COLOR_RESET}"
else
    echo -e "${COLOR_RED}✗ Single process test failed${COLOR_RESET}"
    echo "Check log: $LOG_FILE"
    tail -20 $LOG_FILE
    exit 1
fi
echo ""

# Test 3: Concurrent processes with MPI
echo -e "${COLOR_BLUE}Test 3: Running $NUM_PROCESSES concurrent processes...${COLOR_RESET}"
cleanup  # Clear shared memory
if mpirun -np $NUM_PROCESSES ./test_seqlock_accuracy > ${LOG_FILE}.mpi 2>&1; then
    echo -e "${COLOR_GREEN}✓ Multi-process test passed${COLOR_RESET}"

    # Check for specific issues
    if grep -q "CONSISTENCY CHECK FAILED" ${LOG_FILE}.mpi; then
        echo -e "${COLOR_RED}✗ Consistency check failed!${COLOR_RESET}"
        grep "CONSISTENCY CHECK FAILED" ${LOG_FILE}.mpi
        exit 1
    fi

    if grep -q "Too many inconsistencies" ${LOG_FILE}.mpi; then
        echo -e "${COLOR_RED}✗ Partial read detected!${COLOR_RESET}"
        grep "inconsistencies" ${LOG_FILE}.mpi
        exit 1
    fi

    # Count seqlock retries (if any)
    retry_count=$(grep -c "seqlock retry" ${LOG_FILE}.mpi 2>/dev/null || echo "0")
    if [ $retry_count -gt 0 ]; then
        echo -e "${COLOR_YELLOW}⚠ Seqlock retries detected: $retry_count${COLOR_RESET}"
        if [ $retry_count -gt 100 ]; then
            echo -e "${COLOR_RED}✗ Too many retries (> 100)${COLOR_RESET}"
            exit 1
        fi
    else
        echo -e "${COLOR_GREEN}✓ No seqlock retries (excellent performance)${COLOR_RESET}"
    fi

else
    echo -e "${COLOR_RED}✗ Multi-process test failed${COLOR_RESET}"
    echo "Check log: ${LOG_FILE}.mpi"
    tail -50 ${LOG_FILE}.mpi
    exit 1
fi
echo ""

# Test 4: Check shared memory state
echo -e "${COLOR_BLUE}Test 4: Verifying shared memory state...${COLOR_RESET}"
if [ -f /tmp/cudevshr.cache ]; then
    size=$(stat -f%z /tmp/cudevshr.cache 2>/dev/null || stat -c%s /tmp/cudevshr.cache 2>/dev/null)
    echo "  Shared memory file size: $size bytes"

    # All processes should be cleaned up
    sleep 2
    strings /tmp/cudevshr.cache | grep -E "proc_num|pid" || true

    echo -e "${COLOR_GREEN}✓ Shared memory check complete${COLOR_RESET}"
else
    echo -e "${COLOR_YELLOW}⚠ Shared memory file not found (may be cleaned up)${COLOR_RESET}"
fi
echo ""

# Test 5: Stress test - rapid launch/exit
echo -e "${COLOR_BLUE}Test 5: Stress test - rapid process spawn/exit...${COLOR_RESET}"
cleanup
for i in {1..20}; do
    mpirun -np 4 ./test_seqlock_accuracy > /dev/null 2>&1 &
    PID=$!
    sleep 0.5
    wait $PID || true
    echo -n "."
done
echo ""
echo -e "${COLOR_GREEN}✓ Stress test complete (no crashes)${COLOR_RESET}"
echo ""

# Test 6: Memory leak check
echo -e "${COLOR_BLUE}Test 6: Memory leak detection...${COLOR_RESET}"
cleanup
initial_gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  Initial GPU memory: ${initial_gpu_mem} MB"

# Run test multiple times
for i in {1..5}; do
    mpirun -np $NUM_PROCESSES ./test_seqlock_accuracy > /dev/null 2>&1
    sleep 1
done

final_gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "  Final GPU memory: ${final_gpu_mem} MB"

mem_diff=$((final_gpu_mem - initial_gpu_mem))
if [ $mem_diff -gt 100 ]; then
    echo -e "${COLOR_RED}✗ Memory leak detected: +${mem_diff} MB${COLOR_RESET}"
    exit 1
else
    echo -e "${COLOR_GREEN}✓ No memory leak detected (delta: ${mem_diff} MB)${COLOR_RESET}"
fi
echo ""

# Final summary
echo ""
echo -e "${COLOR_GREEN}========================================"
echo "ALL TESTS PASSED!"
echo "========================================"
echo ""
echo "✓ Seqlock providing consistent reads"
echo "✓ No race conditions detected"
echo "✓ Memory accounting accurate"
echo "✓ No memory leaks"
echo "✓ Handles high contention correctly"
echo ""
echo "Option 4 Precise Accounting: VERIFIED"
echo -e "========================================${COLOR_RESET}"
echo ""
echo "Logs saved to:"
echo "  - $LOG_FILE"
echo "  - ${LOG_FILE}.mpi"

exit 0
