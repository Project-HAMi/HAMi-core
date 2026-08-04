# Owner-death lock recovery benchmark

This directory contains the benchmark source and raw measurements used for the
local `/tmp` comparison discussed in Project-HAMi/HAMi-core#248.

## Scope

This benchmark compares recovery latency after a process holding a one-byte
`fcntl` record lock is killed.

Two acquisition strategies are compared:

- `block`: blocking `F_SETLKW`
- `poll`: nonblocking `F_SETLK` with exponential backoff and jitter modeled
  after the merged HAMi-core #248 strategy

The benchmark uses `/tmp/hami-owner-death.lock`, so these results describe the
tested local filesystem environment only. They should not be generalized to
NFS or other network filesystems.

## Build

    gcc -x c -O2 -Wall -Wextra owner_death_bench.c.txt -o owner_death_bench

## Run

    ./owner_death_bench block
    ./owner_death_bench poll

The published raw data contains 100 samples for each mode.

## Raw data

- `results/owner-block.csv`
- `results/owner-poll.csv`

Columns: `mode,recovery_us,recovery_ms`

The CSV files are a direct structured conversion of the original per-run
benchmark output.
