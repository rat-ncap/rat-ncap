#!/usr/bin/env bash
# Stores the repository's environment variables in the active conda environment (`rat-ncap`).
# Run once after creating the environment, then reactivate it:
#
#   conda activate rat-ncap
#   bash conda-setup.sh [OUTPUT_ROOT]
#   conda deactivate && conda activate rat-ncap
#
# REPO_ROOT: This repository (the directory of this script).
# OUTPUT_ROOT: Where runs write their logs and checkpoints (default: `$REPO_ROOT/outputs`).
# PYTHONPATH: The repository's packages and the project's code.
# MUJOCO_GL: MuJoCo's rendering backend (`egl` for headless GPU rendering; use `osmesa` without a
#   GPU).
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "rat-ncap" ]]; then
  echo "Activate the conda environment first: conda activate rat-ncap" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$(mkdir -p "${1:-$REPO_ROOT/outputs}" && cd "${1:-$REPO_ROOT/outputs}" && pwd)"
PROJECT_ROOT="$REPO_ROOT/projects/rodent"

conda env config vars set \
  REPO_ROOT="$REPO_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PYTHONPATH="$REPO_ROOT/libs:$REPO_ROOT/libs3/tonic:$PROJECT_ROOT:$PROJECT_ROOT/train" \
  MUJOCO_GL="egl"

echo "Set REPO_ROOT=$REPO_ROOT and OUTPUT_ROOT=$OUTPUT_ROOT. Reactivate the environment:"
echo "  conda deactivate && conda activate rat-ncap"
