#!/bin/bash
python -m pytest --cov-report html --cov-config=.coveragerc --cov sigmaepsilon.mesh
