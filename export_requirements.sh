#!/bin/bash
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --only dev --output requirements-dev.txt
poetry export -f requirements.txt --only test --output requirements-test.txt
poetry export -f requirements.txt --only docs --output ./docs/requirements.txt
