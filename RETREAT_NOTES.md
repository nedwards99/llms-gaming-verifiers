# Retreat fork notes — inoculation prompting study

This is the paper's own evaluation code (Helff et al. 2026), used here as the
**measurement** half of an inoculation-prompting study. The training half, the
project overview and all the environment gotchas live in
`../open-instruct-slurm/INOCULATION_PROJECT.md` — **read that first.**

## What we changed

- **`evaluate_model_vllm.py`**
  - `--prompt-variant` / `--paraphrase-idx` / `--variant-position`: applies an
    inoculation instruction to every prompt. The families are **imported** from
    `../open-instruct-slurm/open_instruct/slr/prompt_variants.py` (override
    with `SLR_TRAINING_REPO`) rather than copied, so training and measurement
    cannot drift.
  - `--num-samples`: samples per prompt. Upstream generated one, which makes an
    elicitation rate very noisy.
  - `--dataset-config` / `--dataset-split`: upstream hardcoded `v1-All`/`test`.
  - `--subset-strategy {head,shuffle,stratified}`: SLR-Bench is **ordered by
    tier**, so upstream's `--test-subset N` returns only Basic problems on
    v1-All. `stratified` spreads the subset across tiers.
  - An explicit `--parallel-size` now overrides the per-model heuristics, which
    hardcode tensor-parallel 4 for any `qwen3` model and kill a single-GPU job.
  - `model_outputs.json` now records **both** validation programs.
    `shortcuts.py` needs `validation_program_shortcuts` (extensional) *and*
    `validation_program` (isomorphic); given only the latter it silently falls
    back to a legacy path where both judges agree and the shortcut count is
    always zero.
  - Output tags include the variant, so families don't share a directory.
    (Note: a run is **skipped** if `model_outputs.json` already exists — handy
    for resuming, dangerous if you change `TEST_SUBSET`/`NUM_SAMPLES` between
    runs, since the tag doesn't encode them.)

- **`IPT` submodule** bumped to `origin/main`. The commit the parent repo pins
  exposes a three-argument `verify_ipt` under `IPT/ipt/verifier.py`, while
  `shortcuts.py` calls a five-argument version at `IPT/ipt_verifier.py` — so
  scoring could not even import. Run
  `git submodule update --init --recursive` after cloning.

- **`run_elicitation.sh`** — sweeps all prompt families for one model, then
  scores everything with IPT. Sets up the container environment (writable
  `HOME`, CA bundle, vLLM flags) that the training side needed too. A family
  that crashes is logged and the sweep continues.

- **`submit_elicitation.sh`** — the sbatch wrapper. Holds no settings of its
  own; environment variables pass straight through.

## Running

```bash
sbatch --time=06:00:00 submit_elicitation.sh          # all families, one model
MODEL=Qwen/Qwen3-1.7B sbatch --time=06:00:00 submit_elicitation.sh
```

Knobs: `MODEL`, `FAMILIES`, `TEST_SUBSET`, `NUM_SAMPLES`, `DATASET_CONFIG`,
`OUT_PATH`, `PARALLEL_SIZE`. Budget generously — 5 families × 200 prompts × 4
samples is 4,000 generations and does not fit in two hours.

Re-scoring existing generations needs **no GPU**:

```bash
python shortcuts.py --output-dir output/elicitation
```

Read Table 2 (`Ns = extensional − isomorphic`, per tier) for shortcut counts
and Table 1 for accuracy. Percentages are normalised per 250 problems/tier, so
subset runs understate them — fine for comparing families to each other, not
comparable to the published leaderboard.
