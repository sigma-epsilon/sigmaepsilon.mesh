#!/bin/bash

# Set default rc_path
rc_path=".coveragerc"

# Check if NUMBA_DISABLE_JIT is set, and if not, assign default value of 0
if [[ -z "$NUMBA_DISABLE_JIT" ]]; then
    jit_value=0
else
    jit_value=$NUMBA_DISABLE_JIT
fi

# Check if the first argument is "NUMBA_DISABLE_JIT"
if [[ "$1" == "NUMBA_DISABLE_JIT" ]]; then
    echo "Turning off Numba JIT."
    export NUMBA_DISABLE_JIT=1
    rc_path=".coveragerc_nojit"
fi

# Set the COVERAGE_RCFILE to the chosen rc_path
export COVERAGE_RCFILE=$rc_path

# Run the coverage commands using the specified .coveragerc file
poetry run coverage run --source=src -m pytest tests/
poetry run coverage xml
poetry run coverage html

# Restore the original NUMBA_DISABLE_JIT value
export NUMBA_DISABLE_JIT=$jit_value
