#!/bin/bash
#
# run_dpg_baseline_local.sh
#
# Like run_dpg_local.sh but for the SERVER-LESS baselines
# (levenshtein / embedding_sim / fasttext_lang_id). No metagrad server is launched:
# custom_metagrad_reward.py reads its config from the baseline yaml pointed to by
# METAGRAD_BASELINE_CONFIG and builds+caches the target benchmark in-process from the
# eleuther task named in that yaml.
#
#   - verl PPO trainer    : ./verl, installed editable in the ./.venv-verl venv
#   - custom reward fn     : ./custom_metagrad_reward.py  (baseline mode via METAGRAD_BASELINE_CONFIG)
#   - baseline config      : experiment_configs/lambada_experiments/baselines/<target_metric_type>/baseline_llama3.2_instruct_lambada_<lang>.yaml
#   - verl trainer config  : experiment_configs/lambada_experiments/verl_llama3.2_instruct.yaml (-> hydra overrides via yaml_to_hydra.py)
#   - RL data (copied in)  : ./prompt_data/{wikipedia_{train,test}.parquet, prompt_ending_to_force.txt}
#
# ASSUMPTIONS:
#   - You are already on a node with 8 visible GPUs.
#   - The trainer venv ./.venv-verl has been built (see ./install-verl-env.sh).
#
# Usage: ./run_dpg_baseline_local.sh <experiment_name> <baseline_config> [output_dir]

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <baseline_config> [output_dir]"
    echo "  baseline_config e.g. experiment_configs/lambada_experiments/baselines/levenshtein/baseline_llama3.2_instruct_lambada_es.yaml"
    exit 1
fi

EXPERIMENT_NAME=$1
BASELINE_CONFIG_ARG=$2
OUTPUT_DIR_ARG=${3:-}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
VERL_VENV="$REPO_DIR/.venv-verl"
if [ ! -x "$VERL_VENV/bin/python" ]; then
    echo "Error: trainer venv missing at $VERL_VENV -- build it with ./install-verl-env.sh"; exit 1
fi

# --- Local paths --------------------------------------------------------------
TRAIN_FILES="$REPO_DIR/prompt_data/wikipedia_train.parquet"
VAL_FILES="$REPO_DIR/prompt_data/wikipedia_test.parquet"
PROMPT_ENDING="$REPO_DIR/prompt_data/prompt_ending_to_force.txt"
REWARD_FN="$REPO_DIR/custom_metagrad_reward.py"
BASELINE_CONFIG=$(realpath "$BASELINE_CONFIG_ARG")
VERL_TRAINER_CONFIG="$REPO_DIR/experiment_configs/lambada_experiments/verl_llama3.2_instruct.yaml"

for p in "$TRAIN_FILES" "$VAL_FILES" "$PROMPT_ENDING" "$REWARD_FN" \
         "$BASELINE_CONFIG" "$VERL_TRAINER_CONFIG" "$REPO_DIR/yaml_to_hydra.py" \
         "$REPO_DIR/verl/verl"; do
    [ -e "$p" ] || { echo "Error: required path missing: $p"; exit 1; }
done

# --- Output dir + scratch -----------------------------------------------------
timestamp=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_DIR_ARG" ]; then
    OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR_ARG")
else
    OUTPUT_DIR="$REPO_DIR/experiment_outputs/${EXPERIMENT_NAME}_${timestamp}"
fi
mkdir -p "$OUTPUT_DIR/checkpoints/${EXPERIMENT_NAME}-dpg" "$OUTPUT_DIR/verl_outputs" "$OUTPUT_DIR/reward_outputs"

export LOCAL_FAST_STORAGE="${LOCAL_FAST_STORAGE:-/tmp/$USER}"
mkdir -p "$LOCAL_FAST_STORAGE"

# Tell the reward fn to run server-less in baseline mode + where to dump rollout artifacts.
export METAGRAD_BASELINE_CONFIG="$BASELINE_CONFIG"
export METAGRAD_METADATA_SAVE_DIR="$OUTPUT_DIR/reward_outputs"

echo "Repo:            $REPO_DIR"
echo "Experiment:      $EXPERIMENT_NAME"
echo "Baseline config: $BASELINE_CONFIG"
echo "Output dir:      $OUTPUT_DIR"

# --- Run the verl PPO trainer (no server; reward fn runs baseline mode) -------
source "$VERL_VENV/bin/activate"

export WANDB_PROJECT=dpg_nano
export WANDB_RUN_GROUP=$EXPERIMENT_NAME
export WANDB_JOB_TYPE=rl

echo "Starting verl PPO training (baseline reward, no metagrad server)..."
base_overrides=$(python3 "$REPO_DIR/yaml_to_hydra.py" "$VERL_TRAINER_CONFIG")
# Capture the trainer's real exit code (PIPESTATUS[0], not tee's) so a crash is
# not masked by the pipe. We still attempt the checkpoint merge below regardless,
# but the code is propagated as the script's exit status at the end (see `exit`),
# so a failed run reports FAILED to Slurm/sacct instead of a misleading COMPLETED.
train_rc=0
(
    cd "$REPO_DIR"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
        ${base_overrides} \
        custom_reward_function.path="$REWARD_FN" \
        trainer.experiment_name="${EXPERIMENT_NAME}-verl" \
        trainer.project_name=dpg \
        trainer.default_local_dir="$OUTPUT_DIR/checkpoints/${EXPERIMENT_NAME}-dpg" \
        hydra.run.dir="$OUTPUT_DIR/verl_outputs" \
        2>&1 | tee -a "$OUTPUT_DIR/${EXPERIMENT_NAME}-dpg.log"
    exit "${PIPESTATUS[0]}"
) || train_rc=$?
if [ "$train_rc" -ne 0 ]; then
    echo "Training exited with code $train_rc, but attempting checkpoint merge anyway"
fi

# --- Merge the final FSDP actor checkpoint -----------------------------------
checkpoint_folder=$(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type d ! -path "$OUTPUT_DIR/checkpoints" | head -1)
if [ -f "$checkpoint_folder/latest_checkpointed_iteration.txt" ]; then
    latest_it=$(cat "$checkpoint_folder/latest_checkpointed_iteration.txt")
    actor_dir="$checkpoint_folder/global_step_${latest_it}/actor"
    if [ -d "$actor_dir" ]; then
        python3 "$REPO_DIR/verl/scripts/model_merger.py" merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$OUTPUT_DIR/merged_final_checkpoint"
    else
        echo "Actor directory not found at: $actor_dir"
    fi
else
    echo "No checkpoint iteration file found under $checkpoint_folder"
fi

echo "Done. Outputs in: $OUTPUT_DIR"

# Propagate the trainer's exit code so a crashed run is reported as FAILED
# (not COMPLETED) by Slurm/sacct. A failed checkpoint merge above already aborts
# via `set -e`, so reaching here means only training may have failed.
exit "$train_rc"
