#!/bin/bash
echo "Disabling Numba JIT Compilation"
export NUMBA_DISABLE_JIT=1
poetry run pytest --cov-report=html --cov-config=.coveragerc_nojit --cov=sigmaepsilon.mesh
echo "Enabling Numba JIT Compilation"
export NUMBA_DISABLE_JIT=0