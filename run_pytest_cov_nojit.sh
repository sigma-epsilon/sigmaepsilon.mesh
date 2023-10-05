#!/bin/bash
export NUMBA_DISABLE_JIT=1
python -m pytest --cov-report html --cov-config=.coveragerc_nojit --cov sigmaepsilon.mesh
export NUMBA_DISABLE_JIT=0