#!/bin/bash
echo "Disabling Numba JIT Compilation"
export NUMBA_DISABLE_JIT=1
./run_tests_with_coverage.sh
echo "Enabling Numba JIT Compilation"
export NUMBA_DISABLE_JIT=0