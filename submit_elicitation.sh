#!/bin/bash
# Batch version of run_elicitation.sh -- allocates the node, starts the
# container, and runs the sweep inside it. Holds no sweep settings of its own,
# so the two cannot drift:
#
#   sbatch submit_elicitation.sh
#   MODEL=Qwen/Qwen3-4B TEST_SUBSET=100 NUM_SAMPLES=8 sbatch submit_elicitation.sh
#
# sbatch exports your environment by default, so MODEL/FAMILIES/TEST_SUBSET/
# NUM_SAMPLES/DATASET_CONFIG/OUT_PATH all pass straight through to the sweep.
#
# --mem=64G is host RAM for the pyxis squashfs build on a node that has never
# seen this image; set CONTAINER_IMAGE to a prebuilt .sqsh to skip that.
#SBATCH --job-name=slr-elicitation
#SBATCH --partition=p_csunivie_gres,p_datamining
#SBATCH --account=datamining
# dgx1 is the V100 node (compute capability 7.0): vLLM cannot run bf16
# there and dies with "Bfloat16 is only supported on GPUs with compute
# capability of at least 8.0". Every other node is Ampere or newer.
#SBATCH --exclude=dgx1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/nlp-data/home/users/nicholase99cs/llms-gaming-verifiers}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-helffml/open_instruct_dev:slr}"
CONTAINER_NAME="${CONTAINER_NAME:-open-instruct-slr}"

mkdir -p "${REPO_ROOT}/logs"

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME:-slr-elicitation} (ID: ${SLURM_JOB_ID:-none})"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Image: ${CONTAINER_IMAGE}"
echo "=========================================="

srun --nodes=1 --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/var/spool/slurmd:/var/spool/slurmd" \
  --container-workdir="${REPO_ROOT}" \
  --container-name="${CONTAINER_NAME}" \
  bash "${REPO_ROOT}/run_elicitation.sh"
