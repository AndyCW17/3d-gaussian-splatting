#!/bin/bash
# Runs pytest inside the Docker container, with live source/tests mounted
# so edits are picked up without rebuilding the image. This is the normal
# day-to-day test command for the rest of the project.

MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/tests:/app/tests" \
  splat-project \
  bash -c "pip install pytest -q && PYTHONPATH=src pytest tests/ -v -s"