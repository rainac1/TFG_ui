#!/bin/bash
set -e

CONDA_BASE=/root/miniconda3

source ${CONDA_BASE}/etc/profile.d/conda.sh

conda activate GtsTalkNeRF

exec "$@"
