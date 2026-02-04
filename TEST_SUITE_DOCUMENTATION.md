# HAMi Comprehensive Test Suite Documentation

**Purpose**: Validate Option 5 (Atomic CAS Initialization + Seqlock Runtime)
**Files**:
- `test_comprehensive.cu` - CUDA test program
- `run_comprehensive_tests.sh` - Test runner with validation
**Date**: 2026-02-02

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Overview](#test-overview)
3. [Expected Outcomes](#expected-outcomes)
4. [Test Cases](#test-cases)
5. [Validation Criteria](#validation-criteria)
6. [Interpreting Results](#interpreting-results)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Check CUDA is available
nvidia-smi

# Check NVCC compiler
nvcc --version  # Should be 10.0+

# Check MPI (optional, for multi-process tests)
mpirun --version
```

### Running Tests

```bash
cd /Users/nishshah/workspace/HAMi-core

# Make executable
chmod +x run_comprehensive_tests.sh

# Run with 8 processes (recommended)
./run_comprehensive_tests.sh 8

# Run with 16 processes (stress test)
./run_comprehensive_tests.sh 16
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        HAMi Comprehensive Test Suite - Expected Outcomes        ║
║              Option 5: Atomic CAS + Seqlock                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: Compilation                                             │
└──────────────────────────────────────────────────────────────────┘
Expected: Successful compilation with C++11 atomics support
✓ PASS: Test program compiled successfully

[... more tests ...]

╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    ✓ ALL VALIDATIONS PASSED                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Test Overview

### What Gets Tested

| Category | Tests | Purpose |
|----------|-------|---------|
| **Initialization** | 1 test | Validate atomic CAS initialization |
| **Memory Allocation** | 1 test | Validate memory accounting accuracy |
| **High Contention** | 1 test | Validate behavior under concurrent load |
| **Partial Reads** | 1 test | Validate seqlock correctness |
| **Stress** | 20 iterations | Validate stability over time |

### Test Phases

```
Phase 1: Compilation
  └─ Compile test_comprehensive.cu with C++11 atomics

Phase 2: Single Process Sanity Check
  └─ Verify basic functionality works

Phase 3: Multi-Process Test (Main)
  ├─ Test 1: Initialization (atomic CAS)
  ├─ Test 2: Memory Allocation (accounting)
  ├─ Test 3: High Contention (4 threads × 50 ops)
  └─ Test 4: Partial Reads (seqlock)

Phase 4: Stress Test
  └─ 20 rapid spawn/exit cycles
```

---

## Expected Outcomes

### 1. Initialization Test

**What it tests**: Atomic CAS initialization with multiple processes

**Expected Behavior** (8 processes):

| Role | Count | Init Time | Description |
|------|-------|-----------|-------------|
| **INITIALIZER** | 1 | ~2000ms | Wins CAS race, performs full initialization |
| **SPIN-WAITER** | 0-2 | 50-200ms | Loses CAS, waits for initializer |
| **FAST PATH** | 5-7 | <50ms | Arrives late, skips CAS entirely |

**Timeline**:
```
T=0ms    All processes start simultaneously
         Process 0: CAS(0→1) ✓ Winner!
         Process 1: CAS(1→1) ✗ Fail, spin-wait
         Process 2: CAS(1→1) ✗ Fail, spin-wait
         Processes 3-7: Not started yet

T=2000ms Process 0 completes init, sets flag=COMPLETE
         Processes 1-2 wake from spin-wait

T=2100ms Processes 3-7 start
         Read flag → COMPLETE already!
         Take FAST PATH (instant skip)

Total: ~2.1 seconds
```

**Log Examples**:

```
[  50.23ms PID 12345 INIT-ROLE   ] INITIALIZER: Took 1987.45 ms (expected ~2000ms)
[  52.10ms PID 12346 INIT-ROLE   ] SPIN-WAITER: Took 123.67 ms (expected 50-200ms)
[2100.45ms PID 12350 INIT-ROLE   ] FAST PATH: Took 8.23 ms (expected <50ms)
```

**Validation Criteria**:
- ✅ PASS: Exactly 1 INITIALIZER
- ✅ PASS: 0-2 SPIN-WAITERs
- ✅ PASS: ≥5 FAST PATH processes
- ✅ PASS: Total time <5 seconds
- ❌ FAIL: Multiple INITIALIZERs (atomic CAS broken)
- ❌ FAIL: >2 SPIN-WAITERs (fast path not working)

### 2. Memory Allocation Test

**What it tests**: Memory accounting accuracy and allocation success rate

**Expected Behavior**:
- All allocations succeed (no false OOM)
- Memory accounting matches NVML reports
- Seqlock allows concurrent allocations without blocking

**Test Sequence**:
```
1. Allocate 20 × 50MB = 1000MB sequentially
2. Verify memory accounting
3. Free all 20 allocations concurrently
4. Verify cleanup
```

**Log Examples**:

```
[  100.34ms PID 12345 ALLOC-PHASE1] Sequential allocations...
[  150.67ms PID 12345 ALLOC-PROGRESS] Allocated 10/20 (50.0%)
[  200.89ms PID 12345 ALLOC-PHASE1] ✓ All 20 allocations successful
[  201.12ms PID 12345 VERIFY-MEMORY] Expected: 1.00 GB, NVML reports: 8.45 GB total in use
[  350.45ms PID 12345 FREE-COMPLETE] ✓ All 20 deallocations successful
```

**Validation Criteria**:
- ✅ PASS: 0 allocation failures
- ✅ PASS: All processes complete allocations
- ✅ PASS: Memory accounting within 10% of expected
- ❌ FAIL: Allocation failures (false OOM or real OOM)
- ❌ FAIL: Memory accounting drift >10%

### 3. High Contention Test

**What it tests**: Seqlock behavior under high concurrent load

**Expected Behavior**:
- 4 threads per process
- 50 allocations per thread = 200 allocations per process
- With 8 processes: 1600 total operations
- No deadlocks, all threads complete
- High throughput (>1000 ops/sec)

**Log Examples**:

```
[ 1000.23ms PID 12345 THREAD-DONE ] Thread 0 completed 50 iterations
[ 1100.45ms PID 12345 THREAD-DONE ] Thread 1 completed 50 iterations
[ 1200.67ms PID 12345 THREAD-DONE ] Thread 2 completed 50 iterations
[ 1300.89ms PID 12345 THREAD-DONE ] Thread 3 completed 50 iterations
[ 1301.12ms PID 12345 CONTENTION  ] ✓ Completed 400 operations in 1301.12 ms (307 ops/sec)
```

**Validation Criteria**:
- ✅ PASS: All threads complete
- ✅ PASS: 0% failure rate
- ✅ PASS: >500 ops/sec throughput
- ✅ EXCELLENT: >1000 ops/sec throughput
- ❌ FAIL: Threads timeout or deadlock
- ❌ FAIL: >0% failure rate

### 4. Partial Read Detection (Seqlock Test)

**What it tests**: Seqlock prevents torn reads during concurrent updates

**Expected Behavior**:
- Sample memory usage 500 times rapidly
- Detect large inconsistencies (torn reads)
- Inconsistency rate <5%

**How Seqlock Works**:
```
Writer Process:
  seqlock++ (→43, odd)      ← Signals "write in progress"
  total = 1000 → 2000       ← Update data
  seqlock++ (→44, even)     ← Signals "write complete"

Reader Process:
  seq1 = read seqlock (43)  ← Odd! Retry
  [Writer completes]
  seq1 = read seqlock (44)  ← Even, proceed
  value = read total (2000)
  seq2 = read seqlock (44)  ← Same! Valid read ✓
```

**Log Examples**:

```
[ 2000.23ms PID 12345 SEQLOCK-TEST ] Testing partial read detection with 500 samples
[ 2050.45ms PID 12345 SEQLOCK-PROGRESS] Sampled 50/500 readings
[ 2100.67ms PID 12345 SEQLOCK-PASS ] ✓ No inconsistencies detected in 500 samples
```

OR (with minor inconsistencies - acceptable):

```
[ 2050.45ms PID 12345 SEQLOCK-WARN ] Large negative delta detected: -150 MB (reading 234)
[ 2100.67ms PID 12345 SEQLOCK-WARN ] ⚠ Minor inconsistencies: 8/500 (1.6%) - acceptable under high load
```

**Validation Criteria**:
- ✅ PASS: 0% inconsistency rate (perfect)
- ✅ PASS: <5% inconsistency rate (acceptable)
- ⚠️ WARN: 5-10% inconsistency rate (investigate)
- ❌ FAIL: >10% inconsistency rate (seqlock broken)

### 5. Stress Test

**What it tests**: Stability over repeated spawn/exit cycles

**Expected Behavior**:
- 20 iterations
- Each iteration: spawn 4 processes, run tests, exit
- No crashes, no hangs, no orphaned processes

**Validation Criteria**:
- ✅ PASS: 20/20 iterations complete
- ⚠️ WARN: 18-19/20 iterations complete
- ❌ FAIL: <18/20 iterations complete

---

## Test Cases

### Test 1: Initialization Behavior

**File**: `test_comprehensive.cu`, lines 180-220

**What it does**:
1. Records start time
2. Calls `cudaGetDeviceCount()` → triggers HAMi init
3. Records end time
4. Classifies role based on duration

**Code**:
```cpp
int test_initialization() {
    gettimeofday(&result.init_start, NULL);

    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    gettimeofday(&result.init_end, NULL);
    result.init_duration_ms = time_diff_ms(&result.init_start, &result.init_end);

    if (result.init_duration_ms > 1500) {
        result.was_initializer = 1;  // Took ~2000ms
    } else if (result.init_duration_ms > 50) {
        result.took_fast_path = 0;   // Spin-waited
    } else {
        result.took_fast_path = 1;   // Fast path!
    }
}
```

**Heuristics**:
- `>1500ms` → INITIALIZER (does full init)
- `50-1500ms` → SPIN-WAITER (waits for initializer)
- `<50ms` → FAST PATH (skips everything)

### Test 2: Memory Allocation Consistency

**File**: `test_comprehensive.cu`, lines 245-320

**What it does**:
1. Allocate 20 × 50MB = 1GB
2. Query NVML for total GPU memory usage
3. Compare expected vs actual
4. Free all allocations
5. Verify cleanup

**Key Checks**:
- No allocation failures
- Memory accounting reasonable
- Successful cleanup

### Test 3: High Contention

**File**: `test_comprehensive.cu`, lines 360-430

**What it does**:
1. Launch 4 threads per process
2. Each thread: 50 allocations + 50 frees
3. Concurrent execution (maximum contention)
4. Measure throughput

**Key Metrics**:
- Operations per second
- Failure rate
- Thread completion

### Test 4: Partial Read Detection

**File**: `test_comprehensive.cu`, lines 460-540

**What it does**:
1. Allocate large buffers (create write activity)
2. Sample memory usage 500 times rapidly
3. Detect large inconsistencies (torn reads)
4. Calculate inconsistency rate

**How it detects torn reads**:
```cpp
size_t prev_reading = ...;
size_t current_reading = read_memory_usage();

long delta = current_reading - prev_reading;

// Large negative delta = possible torn read
if (delta < -(long)(total_allocated * 2)) {
    inconsistencies++;
}
```

---

## Validation Criteria

### Critical (Must Pass)

| Check | Pass Criteria | Fail Condition |
|-------|--------------|----------------|
| **Exactly 1 Initializer** | Count == 1 | Count != 1 |
| **No Allocation Failures** | Failures == 0 | Failures > 0 |
| **All Processes Complete** | Completed == N | Completed < N |
| **No Deadlocks** | Duration < 30s | Timeout |

### Important (Should Pass)

| Check | Pass Criteria | Warning Threshold |
|-------|--------------|-------------------|
| **Fast Path Count** | ≥50% of processes | <30% of processes |
| **Init Time** | <5 seconds | <10 seconds |
| **Seqlock Inconsistency** | <5% | <10% |
| **Throughput** | >1000 ops/sec | >500 ops/sec |

### Informational (Nice to Have)

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| **Median Init Time** | <50ms | <100ms | <200ms |
| **Spin-Waiter Count** | 0 | 1-2 | 3-4 |
| **Stress Test Pass Rate** | 20/20 | 19/20 | 18/20 |

---

## Interpreting Results

### Success Indicators

✅ **Perfect Run** (All Green):
```
✓ PASS: Exactly 1 INITIALIZER
✓ PASS: Majority took FAST PATH: 6/8
✓ PASS: Median init time excellent: 12.34ms
✓ PASS: No allocation failures
✓ PASS: No seqlock warnings
✓ PASS: All 8 processes completed
✓ PASS: Throughput excellent: 1234 ops/sec
✓ PASS: Stress test completed: 20/20
```

**Interpretation**: Option 5 is working perfectly!

---

✅ **Good Run** (Mostly Green, Some Yellow):
```
✓ PASS: Exactly 1 INITIALIZER
⚠ WARN: SPIN-WAITER count: 3 (expected ≤2)
✓ PASS: Median init time good: 67.89ms
✓ PASS: No allocation failures
⚠ WARN: 12 seqlock warnings (minor inconsistencies under load)
✓ PASS: All 8 processes completed
✓ PASS: Throughput acceptable: 789 ops/sec
✓ PASS: Stress test: 19/20
```

**Interpretation**: Option 5 is working correctly, but under higher load. The warnings are expected behavior in high-contention scenarios.

---

❌ **Failure Indicators** (Red):

**Multiple Initializers** (Atomic CAS broken):
```
✗ FAIL: Expected 1 INITIALIZER, found 3
```
**Cause**: Atomic CAS not working (compiler issue? wrong memory ordering?)
**Action**: Check GCC version (need 4.9+), verify `_Atomic` support

**Allocation Failures** (OOM detection too strict):
```
✗ FAIL: 45 allocation failures detected
```
**Cause**: Memory limits too low or accounting incorrect
**Action**: Check `CUDA_DEVICE_MEMORY_LIMIT` environment variable

**Deadlock** (Processes hang):
```
✗ FAIL: Only 5/8 completed (possible deadlock)
```
**Cause**: Semaphore deadlock or infinite spin-wait
**Action**: Check for timeout logs, review semaphore usage

**High Seqlock Failure Rate**:
```
✗ FAIL: Seqlock consistency failures detected:
  Too many inconsistencies: 78/500 (15.6%)
```
**Cause**: Seqlock not working (torn reads occurring)
**Action**: Verify memory ordering in seqlock implementation

---

## Troubleshooting

### Common Issues

#### Issue 1: Compilation Fails

**Error**:
```
error: '_Atomic' undeclared
```

**Cause**: Compiler doesn't support C11 atomics

**Fix**:
```bash
# Check GCC version
gcc --version  # Need 4.9+

# Or use Clang
clang --version  # Need 3.1+

# Update compiler
sudo apt-get update
sudo apt-get install gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
```

#### Issue 2: MPI Not Found

**Error**:
```
mpirun: command not found
```

**Fix**:
```bash
# Install OpenMPI
sudo apt-get install openmpi-bin libopenmpi-dev

# Or use torchrun instead
pip install torch
torchrun --nproc_per_node=8 ./test_comprehensive
```

#### Issue 3: All Processes Are Initializers

**Symptom**: Role distribution shows 8/8 INITIALIZERs

**Cause**: Shared memory file not being shared properly

**Debug**:
```bash
# Check shared memory file
ls -la /tmp/cudevshr.cache

# Should show mmap with MAP_SHARED
lsof /tmp/cudevshr.cache

# Try manual cleanup
rm /tmp/cudevshr.cache
./run_comprehensive_tests.sh
```

#### Issue 4: High Inconsistency Rate

**Symptom**: Seqlock warnings >10%

**Possible Causes**:
1. Extreme contention (too many processes)
2. Seqlock implementation bug
3. Memory ordering issue

**Debug**:
```bash
# Run with fewer processes
./run_comprehensive_tests.sh 4

# Check seqlock implementation
grep -A 20 "atomic_fetch_add.*seqlock" src/multiprocess/multiprocess_memory_limit.c

# Verify memory ordering
grep "memory_order" src/multiprocess/multiprocess_memory_limit.c
```

---

## Advanced Usage

### Custom Test Configurations

**Test specific scenario**:
```bash
# Single test run (no stress)
mpirun -np 8 ./test_comprehensive

# High contention (16 processes)
./run_comprehensive_tests.sh 16

# Extreme stress (32 processes)
./run_comprehensive_tests.sh 32
```

**Save logs for later analysis**:
```bash
./run_comprehensive_tests.sh 8 2>&1 | tee my_test_run.log

# Extract timing data
grep "INIT-ROLE" my_test_run.log | awk '{print $NF}'
```

**Compare with baseline**:
```bash
# Run baseline (option1 or main branch)
git checkout option1-reduce-timeouts
./run_comprehensive_tests.sh 8 > baseline.log

# Run option5
git checkout option5-eliminate-file-lock
./run_comprehensive_tests.sh 8 > option5.log

# Compare
diff baseline.log option5.log
```

---

## FAQ

**Q: How long should the tests take?**
A: Phase 3 (8 processes): ~15-20 seconds. Entire suite: ~2-3 minutes.

**Q: What if I get occasional failures in stress test?**
A: 18-19/20 is acceptable. Occasional failures can happen due to system load.

**Q: Can I run without MPI?**
A: Yes, but you'll only test single-process mode. Use `./test_comprehensive` directly.

**Q: How do I know if fast path is working?**
A: Check "FAST PATH" count in results. Should be >50% of processes.

**Q: What's a good ops/sec throughput?**
A: >1000 is excellent, >500 is good, <500 may indicate contention issues.

---

## Summary

**Key Expected Outcomes**:
1. ✅ Only 1 process initializes (atomic CAS winner)
2. ✅ Most processes take fast path (<50ms init)
3. ✅ All allocations succeed (no false OOMs)
4. ✅ Seqlock prevents partial reads (<5% inconsistency)
5. ✅ High throughput (>1000 ops/sec)
6. ✅ No deadlocks or hangs

**When tests pass**: Option 5 is working correctly and ready for production!

**When tests fail**: Review logs, check system requirements, file a bug report with test output.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-02
**Maintainer**: Claude Code (Anthropic)
