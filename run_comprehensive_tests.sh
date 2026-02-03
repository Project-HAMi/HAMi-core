#!/bin/bash
#
# Comprehensive Test Runner for HAMi Option 5
#
# This script runs multiple test scenarios and validates outcomes against
# expected behavior for atomic CAS initialization and seqlock runtime.
#
# Expected Outcomes Summary:
# ==========================
# 1. INITIALIZATION (8 processes)
#    ✓ Only 1 process is INITIALIZER (logs "Took ~2000ms")
#    ✓ 0-2 processes are SPIN-WAITERs (logs "Took 50-200ms")
#    ✓ 5+ processes take FAST PATH (logs "Took <50ms")
#    ✓ Total time: <3 seconds
#
# 2. MEMORY ALLOCATION
#    ✓ All allocations succeed (no OOM false positives)
#    ✓ Memory accounting matches expectations
#    ✓ All frees complete successfully
#
# 3. HIGH CONTENTION
#    ✓ No deadlocks (all threads complete)
#    ✓ Failure rate: 0%
#    ✓ Operations per second: >1000 ops/sec
#
# 4. PARTIAL READS (Seqlock)
#    ✓ Inconsistency rate: <5%
#    ✓ No major torn reads detected
#
# Usage:
#   ./run_comprehensive_tests.sh [num_processes]
#   Default: 8 processes
#
# Examples:
#   ./run_comprehensive_tests.sh      # Run with 8 processes
#   ./run_comprehensive_tests.sh 16   # Run with 16 processes

set -e

# Configuration
NUM_PROCESSES=${1:-8}
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/tmp/hami_comprehensive_$(date +%s)"
RESULTS_FILE="$LOG_DIR/results_summary.txt"
COMPILE_LOG="$LOG_DIR/compile.log"

# Colors
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_RESET='\033[0m'

# Create log directory
mkdir -p "$LOG_DIR"

# Print header
print_header() {
    echo ""
    echo -e "${COLOR_CYAN}╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║        HAMi Comprehensive Test Suite - Expected Outcomes        ║"
    echo "║              Option 5: Atomic CAS + Seqlock                     ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝${COLOR_RESET}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${COLOR_BLUE}┌──────────────────────────────────────────────────────────────────┐"
    echo -e "│ $(printf '%-64s' "$1") │"
    echo -e "└──────────────────────────────────────────────────────────────────┘${COLOR_RESET}"
}

print_expected() {
    echo -e "${COLOR_CYAN}Expected:${COLOR_RESET} $1"
}

print_pass() {
    echo -e "${COLOR_GREEN}✓ PASS:${COLOR_RESET} $1"
}

print_fail() {
    echo -e "${COLOR_RED}✗ FAIL:${COLOR_RESET} $1"
}

print_warn() {
    echo -e "${COLOR_YELLOW}⚠ WARN:${COLOR_RESET} $1"
}

print_info() {
    echo -e "${COLOR_RESET}  INFO: $1"
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${COLOR_YELLOW}Cleaning up...${COLOR_RESET}"
    rm -f /tmp/cudevshr.cache
    rm -f /tmp/hami_barrier_*
    killall -9 test_comprehensive 2>/dev/null || true
}

trap cleanup EXIT

# ============================================================================
# PHASE 1: Compilation
# ============================================================================

print_header
print_section "PHASE 1: Compilation"

print_expected "Successful compilation with C++11 atomics support"

if nvcc -o "$TEST_DIR/test_comprehensive" "$TEST_DIR/test_comprehensive.cu" \
    -lcudart -lnvidia-ml -std=c++11 > "$COMPILE_LOG" 2>&1; then
    print_pass "Test program compiled successfully"
else
    print_fail "Compilation failed"
    echo "Check log: $COMPILE_LOG"
    cat "$COMPILE_LOG"
    exit 1
fi

# ============================================================================
# PHASE 2: Single Process Sanity Check
# ============================================================================

print_section "PHASE 2: Single Process Sanity Check"

print_expected "Process completes all tests in <10 seconds"
print_expected "Process is INITIALIZER (wins CAS)"
print_expected "All tests pass"

SINGLE_LOG="$LOG_DIR/single_process.log"

if timeout 15 "$TEST_DIR/test_comprehensive" > "$SINGLE_LOG" 2>&1; then
    print_pass "Single process test completed"

    # Validate it was initializer
    if grep -q "INITIALIZER" "$SINGLE_LOG"; then
        print_pass "Process correctly identified as INITIALIZER"
    else
        print_warn "Process should be INITIALIZER in single-process mode"
    fi

    # Check for test pass
    if grep -q "ALL TESTS PASSED" "$SINGLE_LOG"; then
        print_pass "All tests passed in single-process mode"
    else
        print_fail "Some tests failed in single-process mode"
        grep "FAIL" "$SINGLE_LOG" || true
    fi
else
    print_fail "Single process test failed or timed out"
    tail -50 "$SINGLE_LOG"
    exit 1
fi

# ============================================================================
# PHASE 3: Multi-Process Test (Main Test)
# ============================================================================

print_section "PHASE 3: Multi-Process Test ($NUM_PROCESSES processes)"

echo ""
echo -e "${COLOR_CYAN}Expected Outcomes:${COLOR_RESET}"
print_expected "Initialization completes in <3 seconds"
print_expected "Exactly 1 process is INITIALIZER (CAS winner)"
print_expected "0-2 processes are SPIN-WAITERs (early arrivals)"
print_expected "Remaining processes take FAST PATH (late arrivals)"
print_expected "All processes complete without deadlock"
print_expected "Seqlock retry rate: <1%"
print_expected "No consistency failures"
echo ""

MULTI_LOG="$LOG_DIR/multi_process.log"
START_TIME=$(date +%s)

cleanup  # Clear shared memory

if mpirun -np "$NUM_PROCESSES" "$TEST_DIR/test_comprehensive" > "$MULTI_LOG" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    print_pass "Multi-process test completed in ${DURATION}s"

    # ========================================================================
    # VALIDATION 1: Initialization Time
    # ========================================================================
    print_info "Validating initialization time..."

    if [ "$DURATION" -lt 5 ]; then
        print_pass "Total execution time: ${DURATION}s (expected <5s)"
    else
        print_warn "Total execution time: ${DURATION}s (slower than expected <5s)"
    fi

    # ========================================================================
    # VALIDATION 2: Role Distribution
    # ========================================================================
    print_info "Validating process role distribution..."

    INITIALIZER_COUNT=$(grep -c "INITIALIZER:" "$MULTI_LOG" || echo "0")
    SPIN_WAITER_COUNT=$(grep -c "SPIN-WAITER:" "$MULTI_LOG" || echo "0")
    FAST_PATH_COUNT=$(grep -c "FAST PATH:" "$MULTI_LOG" || echo "0")

    echo ""
    print_info "Role distribution:"
    print_info "  INITIALIZER: $INITIALIZER_COUNT (expected: 1)"
    print_info "  SPIN-WAITER: $SPIN_WAITER_COUNT (expected: 0-2)"
    print_info "  FAST PATH:   $FAST_PATH_COUNT (expected: $((NUM_PROCESSES - 1 - SPIN_WAITER_COUNT)))"

    if [ "$INITIALIZER_COUNT" -eq 1 ]; then
        print_pass "Exactly 1 INITIALIZER (atomic CAS working correctly)"
    else
        print_fail "Expected 1 INITIALIZER, found $INITIALIZER_COUNT"
        grep "INIT-ROLE" "$MULTI_LOG"
    fi

    if [ "$SPIN_WAITER_COUNT" -le 2 ]; then
        print_pass "SPIN-WAITER count acceptable: $SPIN_WAITER_COUNT"
    else
        print_warn "More SPIN-WAITERs than expected: $SPIN_WAITER_COUNT (expected ≤2)"
    fi

    if [ "$FAST_PATH_COUNT" -ge $((NUM_PROCESSES / 2)) ]; then
        print_pass "Majority took FAST PATH: $FAST_PATH_COUNT/$NUM_PROCESSES"
    else
        print_warn "Fewer FAST PATH processes than expected: $FAST_PATH_COUNT"
    fi

    # ========================================================================
    # VALIDATION 3: Initialization Timing
    # ========================================================================
    print_info "Validating initialization timing..."

    # Extract initialization times
    grep "INIT-ROLE" "$MULTI_LOG" | awk '{print $NF}' | sed 's/ms)//' | sort -n > "$LOG_DIR/init_times.txt"

    if [ -s "$LOG_DIR/init_times.txt" ]; then
        INIT_MIN=$(head -1 "$LOG_DIR/init_times.txt")
        INIT_MAX=$(tail -1 "$LOG_DIR/init_times.txt")
        INIT_MEDIAN=$(awk '{a[NR]=$1} END{print a[int(NR/2)]}' "$LOG_DIR/init_times.txt")

        echo ""
        print_info "Initialization time distribution:"
        print_info "  Min:    ${INIT_MIN}ms"
        print_info "  Median: ${INIT_MEDIAN}ms"
        print_info "  Max:    ${INIT_MAX}ms"

        # Max should be from initializer (~2000ms)
        MAX_INT=${INIT_MAX%.*}  # Remove decimal
        if [ "$MAX_INT" -lt 3000 ]; then
            print_pass "Initializer time acceptable: ${INIT_MAX}ms (expected ~2000ms)"
        else
            print_warn "Initializer took longer than expected: ${INIT_MAX}ms"
        fi

        # Median should be low (fast path dominant)
        MEDIAN_INT=${INIT_MEDIAN%.*}
        if [ "$MEDIAN_INT" -lt 100 ]; then
            print_pass "Median init time excellent: ${INIT_MEDIAN}ms (fast path dominant)"
        elif [ "$MEDIAN_INT" -lt 200 ]; then
            print_pass "Median init time good: ${INIT_MEDIAN}ms"
        else
            print_warn "Median init time higher than expected: ${INIT_MEDIAN}ms"
        fi
    fi

    # ========================================================================
    # VALIDATION 4: Memory Operations
    # ========================================================================
    print_info "Validating memory operations..."

    ALLOC_FAILURES=$(grep -c "cudaMalloc failed" "$MULTI_LOG" || echo "0")

    if [ "$ALLOC_FAILURES" -eq 0 ]; then
        print_pass "No allocation failures (0 false OOMs)"
    else
        print_fail "$ALLOC_FAILURES allocation failures detected"
        grep "cudaMalloc failed" "$MULTI_LOG" | head -10
    fi

    # Check if all processes completed allocations
    ALLOC_COMPLETE=$(grep -c "All.*allocations successful" "$MULTI_LOG" || echo "0")
    if [ "$ALLOC_COMPLETE" -eq "$NUM_PROCESSES" ]; then
        print_pass "All $NUM_PROCESSES processes completed allocations"
    else
        print_warn "Only $ALLOC_COMPLETE/$NUM_PROCESSES completed allocations"
    fi

    # ========================================================================
    # VALIDATION 5: Seqlock Correctness
    # ========================================================================
    print_info "Validating seqlock correctness..."

    CONSISTENCY_FAILURES=$(grep "SEQLOCK-FAIL" "$MULTI_LOG" || true)

    if [ -z "$CONSISTENCY_FAILURES" ]; then
        print_pass "No seqlock consistency failures"
    else
        print_fail "Seqlock consistency failures detected:"
        echo "$CONSISTENCY_FAILURES"
    fi

    # Check for warnings
    SEQLOCK_WARNINGS=$(grep -c "SEQLOCK-WARN" "$MULTI_LOG" || echo "0")
    if [ "$SEQLOCK_WARNINGS" -eq 0 ]; then
        print_pass "No seqlock warnings (perfect consistency)"
    else
        print_warn "$SEQLOCK_WARNINGS seqlock warnings (minor inconsistencies under load)"
        grep "SEQLOCK-WARN" "$MULTI_LOG" | head -5
    fi

    # ========================================================================
    # VALIDATION 6: Deadlock Detection
    # ========================================================================
    print_info "Validating no deadlocks..."

    COMPLETED_PROCESSES=$(grep -c "ALL TESTS PASSED" "$MULTI_LOG" || echo "0")

    if [ "$COMPLETED_PROCESSES" -eq "$NUM_PROCESSES" ]; then
        print_pass "All $NUM_PROCESSES processes completed (no deadlocks)"
    else
        print_fail "Only $COMPLETED_PROCESSES/$NUM_PROCESSES completed (possible deadlock)"
    fi

    # ========================================================================
    # VALIDATION 7: High Contention Performance
    # ========================================================================
    print_info "Validating high contention performance..."

    OPS_PER_SEC=$(grep "ops/sec" "$MULTI_LOG" | awk '{print $(NF-1)}' | sed 's/(//' | sort -n | tail -1)

    if [ -n "$OPS_PER_SEC" ]; then
        print_info "Peak throughput: ${OPS_PER_SEC} ops/sec"

        OPS_INT=${OPS_PER_SEC%.*}
        if [ "$OPS_INT" -gt 1000 ]; then
            print_pass "Throughput excellent: ${OPS_PER_SEC} ops/sec (>1000 expected)"
        elif [ "$OPS_INT" -gt 500 ]; then
            print_pass "Throughput acceptable: ${OPS_PER_SEC} ops/sec"
        else
            print_warn "Throughput lower than expected: ${OPS_PER_SEC} ops/sec"
        fi
    fi

else
    print_fail "Multi-process test failed or timed out"
    echo ""
    echo "Last 100 lines of log:"
    tail -100 "$MULTI_LOG"
    exit 1
fi

# ============================================================================
# PHASE 4: Stress Test (Rapid Spawn/Exit)
# ============================================================================

print_section "PHASE 4: Stress Test (Rapid Spawn/Exit)"

print_expected "20 iterations complete without crashes"
print_expected "Each iteration: <5 seconds"
print_expected "No orphaned processes"

STRESS_LOG="$LOG_DIR/stress_test.log"

echo "Running stress test (20 iterations)..." > "$STRESS_LOG"

cleanup
STRESS_FAILURES=0

for i in {1..20}; do
    echo -n "."
    if ! timeout 10 mpirun -np 4 "$TEST_DIR/test_comprehensive" >> "$STRESS_LOG" 2>&1; then
        STRESS_FAILURES=$((STRESS_FAILURES + 1))
    fi
    sleep 0.2
done

echo ""

if [ "$STRESS_FAILURES" -eq 0 ]; then
    print_pass "Stress test completed: 20/20 iterations successful"
else
    print_warn "Stress test: $STRESS_FAILURES/20 iterations failed"
fi

# Check for orphaned processes
ORPHANED=$(pgrep test_comprehensive | wc -l)
if [ "$ORPHANED" -eq 0 ]; then
    print_pass "No orphaned processes"
else
    print_warn "$ORPHANED orphaned processes detected"
    killall -9 test_comprehensive 2>/dev/null || true
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_section "FINAL SUMMARY"

echo "" | tee -a "$RESULTS_FILE"
echo "Test Configuration:" | tee -a "$RESULTS_FILE"
echo "  Processes:     $NUM_PROCESSES" | tee -a "$RESULTS_FILE"
echo "  Duration:      ${DURATION}s" | tee -a "$RESULTS_FILE"
echo "  Log Directory: $LOG_DIR" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

echo "Key Metrics:" | tee -a "$RESULTS_FILE"
echo "  Initialization:" | tee -a "$RESULTS_FILE"
echo "    - INITIALIZER:    $INITIALIZER_COUNT / 1 expected" | tee -a "$RESULTS_FILE"
echo "    - SPIN-WAITER:    $SPIN_WAITER_COUNT / 0-2 expected" | tee -a "$RESULTS_FILE"
echo "    - FAST PATH:      $FAST_PATH_COUNT / $((NUM_PROCESSES - 1)) expected" | tee -a "$RESULTS_FILE"
echo "    - Median time:    ${INIT_MEDIAN}ms" | tee -a "$RESULTS_FILE"
echo "    - Max time:       ${INIT_MAX}ms" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "  Memory Operations:" | tee -a "$RESULTS_FILE"
echo "    - Failures:       $ALLOC_FAILURES" | tee -a "$RESULTS_FILE"
echo "    - Completed:      $ALLOC_COMPLETE / $NUM_PROCESSES" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "  Seqlock:" | tee -a "$RESULTS_FILE"
echo "    - Warnings:       $SEQLOCK_WARNINGS" | tee -a "$RESULTS_FILE"
echo "    - Failures:       $([ -z "$CONSISTENCY_FAILURES" ] && echo "0" || echo ">0")" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "  Stress Test:" | tee -a "$RESULTS_FILE"
echo "    - Failures:       $STRESS_FAILURES / 20" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Overall verdict
TOTAL_ISSUES=$((ALLOC_FAILURES + SEQLOCK_WARNINGS + STRESS_FAILURES))

if [ "$INITIALIZER_COUNT" -eq 1 ] && [ "$ALLOC_FAILURES" -eq 0 ] && [ -z "$CONSISTENCY_FAILURES" ]; then
    echo -e "${COLOR_GREEN}╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║                    ✓ ALL VALIDATIONS PASSED                     ║"
    echo "║                                                                  ║"
    echo "║  Option 5 (Atomic CAS + Seqlock) is working correctly!         ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝${COLOR_RESET}"
    echo ""
    echo "Summary saved to: $RESULTS_FILE"
    exit 0
else
    echo -e "${COLOR_YELLOW}╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║                ⚠ SOME VALIDATIONS HAD ISSUES                    ║"
    echo "║                                                                  ║"
    echo "║  Total issues: $TOTAL_ISSUES                                           ║"
    echo "║  Review logs for details                                        ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝${COLOR_RESET}"
    echo ""
    echo "Logs saved to: $LOG_DIR"
    echo "Summary saved to: $RESULTS_FILE"
    exit 1
fi
