#!/bin/bash
# Measure how strongly each inoculation-prompt family elicits the enumeration
# shortcut, BEFORE any training. This is the stage that makes the training
# comparison interpretable: families are selected for similar elicitation but
# different linguistic structure, so that differences in inoculation cannot be
# explained by "this prompt just elicited more shortcutting to begin with".
#
# Run from inside the same container as training, on one GPU:
#
# Inference only -- no optimizer states, no gradients -- so the GPU side is
# light. Keep --mem=64G anyway: that is HOST ram, and the first time a node
# builds this image pyxis converts it to squashfs, which is OOM-killed at 32G.
# (Once a node has the image, or once you use a prebuilt .sqsh, 32G is fine.)
#   srun --mem=64G --cpus-per-task=16 --gres=gpu:1 --time=0-01:00:00 --nodes=1 \
#     --partition=p_csunivie_gres,p_datamining --account=datamining \
#     --container-image="helffml/open_instruct_dev:slr" \
#     --container-mounts="/var/spool/slurmd:/var/spool/slurmd" \
#     --container-workdir="/mnt/nlp-data/home/users/nicholase99cs/llms-gaming-verifiers" \
#     --container-name="open-instruct-slr" --pty bash
#
#   MODEL=Qwen/Qwen3-1.7B bash run_elicitation.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# The prompt families live in the training repo and are imported, not copied.
export SLR_TRAINING_REPO="${SLR_TRAINING_REPO:-$(dirname "${REPO_ROOT}")/open-instruct-slurm}"

# Share the training run's caches: the model is likely already downloaded.
CACHE_ROOT="${OI_CACHE_ROOT:-${SLR_TRAINING_REPO}/.cache}"
# $HOME is read-only inside the container and libraries write there at import
# time (flashinfer's JIT workspace, vLLM's usage stats). Same fix as training.
export HOME="${CACHE_ROOT}/home"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TOKENIZERS_PARALLELISM=FALSE
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "${HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

# The login shell exports paths that do not exist in the container.
if [ ! -f "${SSL_CERT_FILE:-}" ]; then
    if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    else
        export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
    fi
fi
unset REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
[ -d "${XDG_RUNTIME_DIR:-}" ] || unset XDG_RUNTIME_DIR

# Qwen3-4B, matching the training script's default. NOTE: Qwen3.5 models are
# NOT loadable in this container -- their architecture is
# Qwen3_5ForConditionalGeneration and the image's vLLM registry knows only up
# to Qwen3NextForCausalLM. Using one needs a newer vLLM, i.e. a new image.
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET_CONFIG="${DATASET_CONFIG:-v1-All}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TEST_SUBSET="${TEST_SUBSET:-200}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
OUT_PATH="${OUT_PATH:-output/elicitation}"
# "neutral" is the baseline, not a variant under study -- it is what the
# trained models are later evaluated on.
FAMILIES="${FAMILIES:-neutral scope_narrow scope_broad permission goal_redefinition}"
EXTRA_ARGS="${EXTRA_ARGS:---enable-thinking}"
# One GPU unless told otherwise. Without this the script's model heuristics
# pick a tensor-parallel size sized for the authors' 8-GPU nodes.
PARALLEL_SIZE="${PARALLEL_SIZE:-1}"

echo "============================================================"
echo "Elicitation sweep"
echo "  model:    ${MODEL}"
echo "  data:     SLR-Bench ${DATASET_CONFIG}/${DATASET_SPLIT}, ${TEST_SUBSET} prompts x ${NUM_SAMPLES} samples"
echo "  families: ${FAMILIES}"
echo "  out:      ${OUT_PATH}"
echo "============================================================"

# One family's crash must not take the rest of the sweep with it: job 697216
# died on a CUDA device-side assert partway through and lost the three
# families that had not run yet. Failures are recorded and the loop continues.
failed_families=""
for family in ${FAMILIES}; do
    echo
    echo "--- ${family} ---"
    python evaluate_model_vllm.py \
        --model "${MODEL}" \
        --dataset-config "${DATASET_CONFIG}" \
        --dataset-split "${DATASET_SPLIT}" \
        --test-subset "${TEST_SUBSET}" \
        --subset-strategy "${SUBSET_STRATEGY:-stratified}" \
        --num-samples "${NUM_SAMPLES}" \
        --prompt-variant "${family}" \
        --parallel-size "${PARALLEL_SIZE}" \
        --out-path "${OUT_PATH}" \
        ${EXTRA_ARGS} || { echo "WARNING: family ${family} FAILED, continuing"; failed_families="${failed_families} ${family}"; }
done

if [ -n "${failed_families}" ]; then
    echo
    echo "!!! families that failed:${failed_families}"
fi

echo
echo "--- scoring all families with IPT ---"
python shortcuts.py --output-dir "${OUT_PATH}"
echo
echo "Per-family results under ${OUT_PATH}/ipt_results/"
