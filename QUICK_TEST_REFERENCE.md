# Quick Test Reference Card

**Option 5: Atomic CAS + Seqlock**

---

## Run Tests

```bash
./run_comprehensive_tests.sh 8
```

---

## Expected Outcomes (8 Processes)

### ✅ Initialization

| Metric | Expected Value | Meaning |
|--------|----------------|---------|
| **INITIALIZER count** | 1 | Only one process wins CAS |
| **SPIN-WAITER count** | 0-2 | Processes 1-2 may wait briefly |
| **FAST PATH count** | 5-7 | Most processes skip initialization |
| **Total time** | <5 seconds | Fast startup |
| **Initializer time** | ~2000ms | Does full GPU enumeration |
| **Spin-waiter time** | 50-200ms | Waits for initializer |
| **Fast path time** | <50ms | Instant skip |

### ✅ Memory Operations

| Metric | Expected Value | Meaning |
|--------|----------------|---------|
| **Allocation failures** | 0 | No false OOMs |
| **Completed processes** | 8/8 | All finish successfully |
| **Memory accounting** | Within 10% | Accurate tracking |

### ✅ High Contention

| Metric | Expected Value | Meaning |
|--------|----------------|---------|
| **Thread completion** | 100% | No deadlocks |
| **Failure rate** | 0% | All operations succeed |
| **Throughput** | >1000 ops/sec | Excellent performance |

### ✅ Seqlock (Partial Reads)

| Metric | Expected Value | Meaning |
|--------|----------------|---------|
| **Inconsistency rate** | <5% | Seqlock working correctly |
| **Warnings** | 0-20 | Minor inconsistencies acceptable |
| **Failures** | 0 | No major torn reads |

### ✅ Stress Test

| Metric | Expected Value | Meaning |
|--------|----------------|---------|
| **Pass rate** | 18-20/20 | Stable over time |
| **Orphaned processes** | 0 | Clean shutdown |

---

## Visual Guide

### Good Result Example

```
┌──────────────────────────────────────────────────────┐
│ PHASE 3: Multi-Process Test (8 processes)          │
└──────────────────────────────────────────────────────┘

Expected: Exactly 1 process is INITIALIZER (CAS winner)
✓ PASS: Exactly 1 INITIALIZER (atomic CAS working correctly)

Expected: 0-2 processes are SPIN-WAITERs (early arrivals)
✓ PASS: SPIN-WAITER count acceptable: 1

Expected: Remaining processes take FAST PATH (late arrivals)
✓ PASS: Majority took FAST PATH: 6/8

Expected: Initialization completes in <3 seconds
✓ PASS: Total execution time: 2s (expected <5s)

Expected: All allocations succeed (no OOM false positives)
✓ PASS: No allocation failures (0 false OOMs)

Expected: Seqlock retry rate: <1%
✓ PASS: No seqlock warnings (perfect consistency)

Expected: All processes complete without deadlock
✓ PASS: All 8 processes completed (no deadlocks)

Expected: Operations per second: >1000 ops/sec
✓ PASS: Throughput excellent: 1234 ops/sec (>1000 expected)

╔══════════════════════════════════════════════════════╗
║                                                      ║
║            ✓ ALL VALIDATIONS PASSED                 ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### Warning Example (Still Acceptable)

```
⚠ WARN: SPIN-WAITER count: 3 (expected ≤2)
✓ PASS: Majority took FAST PATH: 4/8
⚠ WARN: 12 seqlock warnings (minor inconsistencies under load)
✓ PASS: All 8 processes completed (no deadlocks)
```

**Interpretation**: System under higher load, but still working correctly.

### Failure Example

```
✗ FAIL: Expected 1 INITIALIZER, found 3
```

**Action**: Atomic CAS broken. Check compiler version (need GCC 4.9+).

---

## Timing Cheat Sheet

### Process Initialization Times

```
INITIALIZER:   ████████████████████ ~2000ms (Full init)
SPIN-WAITER:   ███ ~100ms           (Brief wait)
FAST PATH:     █ <50ms              (Instant skip)
```

### By Process Rank (Typical)

```
Rank 0: ████████████████████ 2000ms (INITIALIZER - First started)
Rank 1: ███                   120ms (SPIN-WAITER - Lost CAS)
Rank 2: ███                   115ms (SPIN-WAITER - Lost CAS)
Rank 3: █                      35ms (FAST PATH - Late arrival)
Rank 4: █                      28ms (FAST PATH - Late arrival)
Rank 5: █                      22ms (FAST PATH - Late arrival)
Rank 6: █                      18ms (FAST PATH - Late arrival)
Rank 7: █                      15ms (FAST PATH - Late arrival)
```

---

## Quick Validation Checklist

**After running tests, verify:**

- [ ] Compilation succeeded
- [ ] Exactly 1 INITIALIZER found
- [ ] At least 50% processes took FAST PATH
- [ ] Total time <5 seconds
- [ ] 0 allocation failures
- [ ] Seqlock inconsistency rate <5%
- [ ] All processes completed (no deadlock)
- [ ] Throughput >500 ops/sec (>1000 = excellent)
- [ ] Stress test pass rate ≥18/20

**If all checked**: Option 5 is working correctly! ✅

---

## Interpreting the Role Distribution

### Perfect Distribution (Rank = Launch Order)

```
Process Rank 0: INITIALIZER  ← First to start, wins CAS
Process Rank 1: SPIN-WAITER  ← Second, loses CAS, waits
Process Rank 2: FAST PATH    ← Third, arrives late
Process Rank 3: FAST PATH    ← Fourth, arrives late
Process Rank 4: FAST PATH    ← Fifth, arrives late
Process Rank 5: FAST PATH    ← Sixth, arrives late
Process Rank 6: FAST PATH    ← Seventh, arrives late
Process Rank 7: FAST PATH    ← Eighth, arrives late
```

**Why this is good**: Test script staggers starts (rank × 10ms), so later ranks should take fast path.

### Good Distribution (Some Variation)

```
1 INITIALIZER  ← One process did initialization
2 SPIN-WAITERs ← Two processes arrived early, waited
5 FAST PATHs   ← Five processes arrived late, skipped
```

**Why this is acceptable**: System timing variations are normal.

### Bad Distribution (Fast Path Not Working)

```
1 INITIALIZER
7 SPIN-WAITERs  ← Everyone waiting! Fast path broken!
0 FAST PATHs
```

**Action**: Check atomic read implementation in fast path check.

---

## Performance Thresholds

### Initialization Time (8 processes)

| Time | Grade | Status |
|------|-------|--------|
| <3s | A+ | Excellent - Optimal performance |
| 3-5s | A | Good - Expected performance |
| 5-8s | B | Acceptable - Some contention |
| 8-12s | C | Slow - Investigate contention |
| >12s | F | Failure - Fast path not working |

### Throughput (ops/sec)

| Rate | Grade | Status |
|------|-------|--------|
| >1500 | A+ | Excellent - Minimal contention |
| 1000-1500 | A | Good - Expected performance |
| 500-1000 | B | Acceptable - Moderate contention |
| 200-500 | C | Slow - High contention |
| <200 | F | Failure - Serialization issues |

### Seqlock Inconsistency Rate

| Rate | Grade | Status |
|------|-------|--------|
| 0% | A+ | Perfect - No retries needed |
| <1% | A | Excellent - Rare retries |
| 1-5% | B | Good - Acceptable retries |
| 5-10% | C | High - Investigate load |
| >10% | F | Failure - Torn reads occurring |

---

## Troubleshooting Quick Guide

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Multiple INITIALIZERs | Atomic CAS broken | Check GCC ≥4.9, verify C11 support |
| No FAST PATH | Staggering not working | Check test script delays |
| Allocation failures | Memory limit too low | Increase `CUDA_DEVICE_MEMORY_LIMIT` |
| High inconsistency | Too much contention | Run with fewer processes |
| Deadlock | Semaphore issue | Check semaphore timeout logs |
| Compilation error | Missing atomics | Install GCC 4.9+ or Clang 3.1+ |

---

## Log File Locations

After test run, logs saved to `/tmp/hami_comprehensive_[timestamp]/`:

```
compile.log          - Compilation output
single_process.log   - Single process test
multi_process.log    - Main multi-process test
stress_test.log      - Stress test iterations
init_times.txt       - Extracted initialization times
results_summary.txt  - Final summary
```

---

## One-Line Validation

```bash
# Quick pass/fail check
./run_comprehensive_tests.sh 8 && echo "✅ PASS" || echo "❌ FAIL"
```

---

**Keep this card handy when running tests!**

For detailed explanations, see `TEST_SUITE_DOCUMENTATION.md`
