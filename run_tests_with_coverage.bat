@echo off

set rc_path = .coveragerc

if defined NUMBA_DISABLE_JIT (
    set jit_value=%NUMBA_DISABLE_JIT%
) else (
    set jit_value=0
)

if "%1"=="NUMBA_DISABLE_JIT" (
    echo Turning off Numba JIT.
    set NUMBA_DISABLE_JIT=1
    set rc_path = .coveragerc_nojit
)

set COVERAGE_RCFILE=%rc_path%

uv run coverage run --source=src -m pytest tests/
uv run coverage xml
uv run coverage html

set NUMBA_DISABLE_JIT=%jit_value%
