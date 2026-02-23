#!/bin/bash
rsync -av --delete --exclude='__pycache__' \
  /home/mike/workspace/github/pinescript/pynecore/.venv/lib/python3.13/site-packages/pynecore/ \
  /home/mike/workspace/github/pynecore/src/pynecore/
