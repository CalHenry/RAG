#!/bin/bash
set -e

# Add data and models folders
mkdir -p \
  data/raw \
  data/interim \
  data/processed \
  data/database \
  models

echo "create missing directories:
data/raw
data/interim
data/processed
data/database
models
"
